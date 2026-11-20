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
from prepare_local import DEFAULT_CONFIG, connect_groups, prepare, select_candidates, verify_run
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
        self.config = {**DEFAULT_CONFIG, "include_classes": ["A", "B"], "min_side": 4,
                       "max_images_per_class": 8, "min_class_images": 3, "image_size": 16,
                       "visual_audit_groups_per_class": 1}

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

    def test_selection_reserves_visual_groups_and_respects_products(self):
        full, selected = select_candidates(self.audit, self.config)
        visual = full.loc[full.exclusion_reason.eq("visual_audit_group")]
        self.assertTrue(set(visual.product_code).isdisjoint(selected.product_code))
        self.assertTrue(selected.groupby("label").size().le(8).all())
        for product in selected.product_code.unique():
            self.assertEqual(sum(selected.product_code.eq(product)), sum(self.audit.product_code.eq(product)))
        again = select_candidates(self.audit.sample(frac=1, random_state=3), self.config)[1]
        self.assertEqual(set(selected.article_id), set(again.article_id))

    def test_conflicting_labels_exclude_whole_group(self):
        audit = self.audit.copy()
        audit.loc[0, "product_type_name"] = "B"
        full, selected = select_candidates(audit, self.config)
        self.assertTrue(full.loc[[0, 1], "exclusion_reason"].eq("conflicting_group_labels").all())
        self.assertFalse(selected.product_code.eq("0000000").any())

    def test_deduplication_preserves_transitive_origin_component(self):
        audit = self.audit.copy()
        audit.loc[2, "pixel_sha256"] = audit.loc[0, "pixel_sha256"]
        cfg = {**self.config, "max_images_per_class": 100, "visual_audit_groups_per_class": 0}
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
        self.assertEqual(len(usable), 16)
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
        self.assertEqual(verify_run(run, self.root).split.ne("excluded").sum(), 16)


if __name__ == "__main__":
    unittest.main()
