import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/data"))
from data_contract import publish
from process_data import inspect_image
from prepare_local import DEFAULT_CONFIG, apply_visual_review, connect_groups, prepare, select_candidates, verify_run
from local_dataset import LocalImageDataset
from audit_images import inspect_record


class LocalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "data/raw/images"
        self.raw.mkdir(parents=True)
        rows = []
        for i in range(40):
            article = f"{i:010d}"
            image = Image.new("RGB", (12+i%3, 20), (i*5, i*3, i))
            image.putpixel((1, 1), (255, 255, 255))
            path = self.raw / (article + ".png")
            image.save(path)
            rows.append({"article_id": article, "product_code": f"{i//2:07d}",
                         "product_type_name": "A" if i < 20 else "B", "raw_path": path.name,
                         "image_exists": True, "file_bytes": path.stat().st_size, **inspect_image(path)})
        self.audit = pd.DataFrame(rows)
        publish(self.root / "data/audit", {"image_audit": self.audit},
                {"summary": {"raw_image_base": str(self.raw), "raw_bytes": int(self.audit.file_bytes.sum())}})
        self.config = {**DEFAULT_CONFIG, "include_classes": ["A", "B"], "include_product_groups": None, "min_side": 4,
                       "min_class_images": 3, "image_size": 16}

    def tearDown(self):
        self.temp.cleanup()

    def test_transitive_groups_connect_raw_and_prepared_duplicates(self):
        frame = pd.DataFrame({"article_id": ["1", "2", "3", "4"], "product_code": ["a", "a", "b", "c"],
                              "raw": ["x", "y", "y", "z"], "prepared": ["r", "s", "t", "t"]})
        self.assertEqual(connect_groups(frame, ["raw", "prepared"]).nunique(), 1)

    def test_audit_inspector_matches_reference_modes_orientation_and_corruption(self):
        for mode in ["RGB", "L", "RGBA"]:
            path = self.raw / (mode + ".png")
            Image.new(mode, (9, 7)).save(path)
            result = inspect_record(({"raw_path": path.name}, str(self.raw)))
            expected = inspect_image(path)
            self.assertEqual({k: result[k] for k in expected}, expected)
        path = self.raw / "rotated.jpg"
        image = Image.new("RGB", (12, 20), "red")
        exif = Image.Exif(); exif[274] = 6
        image.save(path, exif=exif)
        result = inspect_record(({"raw_path": path.name}, str(self.raw)))
        expected = inspect_image(path)
        self.assertEqual({k: result[k] for k in expected}, expected)
        path.write_bytes(b"not an image")
        self.assertFalse(inspect_record(({"raw_path": path.name}, str(self.raw)))["image_valid"])

    def test_selection_keeps_all_eligible_products_without_equalizing_classes(self):
        audit = self.audit.iloc[:32].copy()
        full, selected = select_candidates(audit, self.config)
        self.assertEqual(selected.groupby('label').size().to_dict(), {'A': 20, 'B': 12})
        self.assertFalse(full.exclusion_reason.isin(['class_budget', 'visual_audit_group']).any())
        for product in selected.product_code.unique():
            self.assertEqual(sum(selected.product_code.eq(product)), sum(audit.product_code.eq(product)))
        again = select_candidates(audit.sample(frac=1, random_state=3), self.config)[1]
        self.assertEqual(set(selected.article_id), set(again.article_id))

    def test_exposure_anchors_keep_development_products_out_of_test(self):
        anchors = self.audit.loc[[0, 2, 4, 20, 22, 24], ['article_id']].copy()
        anchors['split'] = ['train', 'validation', 'test'] * 2
        anchors.to_csv(self.root/'exposure.csv', index=False)
        cfg = {**self.config, 'split_exposure_path': 'exposure.csv'}
        run = prepare(cfg, self.root, workers=2)
        frame = verify_run(run, self.root).set_index('article_id')
        for row in anchors.itertuples():
            self.assertEqual(frame.loc[row.article_id, 'split'], row.split)

    def test_family_scope_keeps_every_type_and_detects_an_incomplete_snapshot(self):
        audit = self.audit.copy()
        audit['product_group_name'] = np.where(audit.product_type_name.eq('A'), 'Garment Upper body', 'Shoes')
        cfg = {**self.config, 'include_product_groups': ['Garment Upper body'], 'include_classes': ['A']}
        _, selected = select_candidates(audit, cfg)
        self.assertEqual(len(selected), 20)
        self.assertTrue(selected.product_type_name.eq('A').all())
        audit.loc[0, 'product_type_name'] = 'C'
        with self.assertRaisesRegex(ValueError, 'todos los tipos'):
            select_candidates(audit, cfg)

    def test_merged_exposure_components_use_most_exposed_partition(self):
        from prepare_local import exposure_by_group
        frame = pd.DataFrame({'group_id': ['a','a','b','b'],
                              'history_split': ['test','train','test','validation']})
        self.assertEqual(exposure_by_group(frame).tolist(), ['train','train','validation','validation'])

    def test_conflicting_labels_exclude_whole_group(self):
        audit = self.audit.copy()
        audit.loc[0, "product_type_name"] = "B"
        full, selected = select_candidates(audit, self.config)
        self.assertTrue(full.loc[[0, 1], "exclusion_reason"].eq("conflicting_group_labels").all())
        self.assertFalse(selected.product_code.eq("0000000").any())

    def test_visual_review_excludes_only_confirmed_training_image(self):
        candidate = self.audit.copy()
        candidate['label'] = candidate.product_type_name
        candidate['split'] = ['train'] * 36 + ['validation'] * 2 + ['test'] * 2
        row = candidate.iloc[0]
        review = pd.DataFrame([{'article_id': row.article_id, 'file_sha256': row.file_sha256,
                                'decision': 'exclude_detail', 'reason': 'Only fabric detail visible'}])
        result, report = apply_visual_review(candidate, review)
        self.assertEqual(result.loc[0, 'split'], 'excluded')
        self.assertEqual(result.loc[0, 'assigned_split'], 'train')
        pd.testing.assert_frame_equal(result.loc[1:, candidate.columns], candidate.loc[1:])
        self.assertEqual(report['excluded_detail_images'], 1)
        self.assertEqual(report['unreviewed_images'], 39)
        self.assertFalse(report['complete_visual_review'])

    def test_visual_review_filters_validation_and_test_and_rejects_changed_original(self):
        candidate = self.audit.copy()
        candidate['label'] = candidate.product_type_name
        candidate['split'] = ['train'] * 36 + ['validation'] * 2 + ['test'] * 2
        for index in [36, 38]:
            row = candidate.iloc[index]
            review = pd.DataFrame([{'article_id': row.article_id, 'file_sha256': row.file_sha256,
                                    'decision': 'exclude_detail', 'reason': 'Detail'}])
            result, _ = apply_visual_review(candidate, review)
            self.assertEqual(result.loc[index, 'split'], 'excluded')
            self.assertEqual(result.loc[index, 'assigned_split'], candidate.loc[index, 'split'])
        review.loc[0, 'article_id'] = candidate.iloc[0].article_id
        review.loc[0, 'file_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'foto revisada ha cambiado'):
            apply_visual_review(candidate, review)

    def test_visual_review_publication_recomputes_training_statistics(self):
        previous = prepare(self.config, self.root, workers=2)
        before = verify_run(previous, self.root)
        row = before.loc[before.split.eq('train')].iloc[0]
        review = pd.DataFrame([{'article_id': row.article_id, 'file_sha256': row.file_sha256,
                                'decision': 'exclude_detail', 'reason': 'Detail confirmed by reviewer'}])
        review.to_csv(self.root / 'review.csv', index=False)
        cfg = {**self.config, 'training_visual_review_path': 'review.csv'}
        current = prepare(cfg, self.root, workers=2)
        after = verify_run(current, self.root)
        rejected = after.loc[after.article_id.eq(row.article_id)].iloc[0]
        self.assertEqual(rejected.reason, 'detail_only_image')
        self.assertEqual(rejected.training_weight, 0)
        for split in ['validation', 'test']:
            self.assertEqual(set(before.loc[before.split.eq(split), 'article_id']),
                             set(after.loc[after.split.eq(split), 'article_id']))
        stats = json.loads((current / 'normalization.json').read_text())
        self.assertEqual(stats['pixels'], int(after.split.eq('train').sum()) * cfg['image_size']**2)
        self.assertEqual(prepare(cfg, self.root, workers=2), current)

    def test_deduplication_preserves_transitive_origin_component(self):
        audit = self.audit.copy()
        audit.loc[2, "pixel_sha256"] = audit.loc[0, "pixel_sha256"]
        cfg = self.config
        full, selected = select_candidates(audit, cfg)
        self.assertEqual(full.loc[2, "exclusion_reason"], "exact_pixel_duplicate")
        self.assertEqual(selected.loc[0, "raw_group_id"], selected.loc[3, "raw_group_id"])
        selected["prepared_pixel_sha256"] = selected.article_id
        groups = connect_groups(selected, ["raw_group_id", "pixel_sha256", "prepared_pixel_sha256"])
        self.assertEqual(groups.loc[0], groups.loc[3])

    def test_publication_is_portable_and_works_without_raw(self):
        run = prepare(self.config, self.root, workers=2)
        manifest = verify_run(run, self.root)
        usable = manifest.loc[manifest.split.ne("excluded")]
        self.assertEqual(len(usable), 40)
        self.raw.rename(self.raw.with_name("images.offline"))
        self.assertEqual(prepare(self.config, self.root, workers=2), run)
        dataset = LocalImageDataset("train", self.root)
        batch, labels = next(dataset.batches(3))
        self.assertEqual(batch.shape, (3, 3, 16, 16))
        self.assertTrue(np.isfinite(batch).all())
        self.assertEqual(labels.dtype, np.int64)
        with self.assertRaisesRegex(ValueError, "solo"):
            LocalImageDataset("test", self.root, augment=True)

    def test_statistics_use_training_pixels_only(self):
        run = prepare(self.config, self.root, workers=2)
        dataset = LocalImageDataset("train", self.root, normalize=False)
        images = np.stack([dataset[i][0] for i in range(len(dataset))])
        stats = json.loads((run / "normalization.json").read_text())
        np.testing.assert_allclose(stats["mean"], images.astype("float64").mean(axis=(0, 2, 3)), atol=1e-7)
        np.testing.assert_allclose(stats["std"], images.astype("float64").std(axis=(0, 2, 3)), atol=1e-7)

    def test_changed_raw_does_not_replace_current(self):
        run = prepare(self.config, self.root, workers=2)
        pointer = self.root / "data/processed/CURRENT.json"
        before = pointer.read_bytes()
        selected = select_candidates(self.audit, self.config)[1]
        (self.raw / selected.iloc[0].raw_path).write_bytes(b"changed")
        changed = {**self.config, "jpeg_quality": 91}
        with self.assertRaisesRegex(ValueError, "RAW modificado"):
            prepare(changed, self.root, workers=2)
        self.assertEqual(before, pointer.read_bytes())
        self.assertEqual(verify_run(run, self.root).split.ne("excluded").sum(), 40)


if __name__ == "__main__":
    unittest.main()
