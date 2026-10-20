import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/data"))
import data_contract as common
import process_data as vision

class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_duplicate_names_rejected(self):
        Image.new("RGB", (3, 3)).save(self.directory / "001.png")
        Image.new("RGB", (3, 3)).save(self.directory / "001.jpg")
        with self.assertRaisesRegex(ValueError, "duplicado"):
            vision.image_index(self.directory)

    def test_corrupt_and_truncated_images(self):
        path = self.directory / "bad.jpg"
        path.write_bytes(b"not an image")
        self.assertFalse(vision.inspect_image(path)["image_valid"])
        Image.new("RGB", (100, 100), "red").save(path)
        path.write_bytes(path.read_bytes()[:-40])
        self.assertFalse(vision.inspect_image(path)["image_valid"])

    def test_pixel_hash_and_channels(self):
        a, b = self.directory / "a.png", self.directory / "b.png"
        Image.new("L", (10, 10), 100).save(a)
        Image.new("RGB", (10, 10), (100, 100, 100)).save(b)
        self.assertEqual(vision.inspect_image(a)["pixel_sha256"], vision.inspect_image(b)["pixel_sha256"])

    def test_group_integrity_reproducibility_and_weights(self):
        rows = []
        for i in range(24):
            article = f"{i:010d}"
            Image.new("RGB", (6, 6), (i * 10, i, 0)).save(self.directory / (article + ".png"))
            rows.append({"article_id": article, "product_code": f"{i // 2:07d}",
                         "garment_group_name": "A" if i < 12 else "B"})
        # Dos imágenes del mismo producto tienen además los mismos píxeles.
        (self.directory / "0000000001.png").write_bytes((self.directory / "0000000000.png").read_bytes())
        articles = pd.DataFrame(rows)
        args = (articles, self.directory, "garment_group_name", 42, .15, .15)
        manifest, labels, weights, summary = vision.prepare_manifest(*args)
        again, _, _, _ = vision.prepare_manifest(*args)
        pd.testing.assert_frame_equal(manifest, again)
        for col in ["product_code", "pixel_sha256", "group_id"]:
            self.assertTrue(manifest.groupby(col).split.nunique().eq(1).all())
        self.assertEqual(set(labels), {"A", "B"})
        self.assertEqual(set(weights), {"A", "B"})
        self.assertEqual(summary["classes_usable"], 2)

    def test_unfeasible_group_split(self):
        frame = pd.DataFrame({"label": ["A"] * 3, "group_id": ["a", "b", "c"],
                              "product_code": ["a", "b", "c"], "pixel_sha256": ["a", "b", "c"]})
        with self.assertRaisesRegex(ValueError, "no permiten"):
            vision.grouped_splits(frame, 42, .49, .49)

    def test_no_images_never_produces_fake_split(self):
        articles = pd.DataFrame({"article_id": ["001"], "product_code": ["01"], "garment_group_name": ["A"]})
        with self.assertRaisesRegex(ValueError, "No hay imágenes"):
            vision.prepare_manifest(articles, self.directory, "garment_group_name", 42, .15, .15)

class PublicationTests(unittest.TestCase):
    def test_atomic_publication_and_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            first = common.publish(out, {"test": pd.DataFrame({"id": ["001"]})}, {"summary": {"ok": True}})
            before = (out / "CURRENT.json").read_bytes()
            with patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    common.publish(out, {"test": pd.DataFrame({"id": ["002"]})}, {})
            self.assertEqual(before, (out / "CURRENT.json").read_bytes())
            common.publish(out, {}, {"summary": {"preview": True}}, preview=True)
            self.assertEqual(before, (out / "CURRENT.json").read_bytes())
            self.assertTrue((first / "checksums.json").is_file())

    def test_identifier_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "articles.csv"
            row = {col: "A" for col in common.ARTICLE_COLUMNS}
            row.update(article_id="00001", product_code="0001")
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertEqual(common.load_articles(path).article_id.iloc[0], "00001")
            row["article_id"] = ""
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                common.load_articles(path)

if __name__ == "__main__":
    unittest.main()
