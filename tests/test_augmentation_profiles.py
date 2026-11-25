"""Perfiles reales con una imagen diminuta; sin preparación ni entrenamiento."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from src.data.local_dataset import LocalImageDataset, AUGMENTATION_PROFILES


class AugmentationProfileTests(unittest.TestCase):
    def test_profiles_change_pixels_reproducibly_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'data/processed/demo'
            run.mkdir(parents=True)
            Image.new('RGB', (4,4), (150,70,25)).save(run/'image.png')
            original = (run/'image.png').read_bytes()
            (run/'image_manifest.csv.gz').write_bytes(b'')
            import pandas as pd
            pd.DataFrame([{'article_id':'001', 'split':'train', 'image_path':'demo/image.png', 'label_id':0}]).to_csv(run/'image_manifest.csv.gz',index=False)
            (run/'config.json').write_text(json.dumps({'image_size':4}))
            (run/'normalization.json').write_text(json.dumps({'mean':[0,0,0], 'std':[1,1,1]}))
            results = {}
            for profile, strength in [('suave',.1),('fuerte',.5)]:
                reader = LocalImageDataset('train',root,run,augment=True,normalize=False,augmentation_profile=profile)
                self.assertEqual(reader.augmentation_config['brightness'],strength)
                self.assertEqual(reader.augmentation_config['colour'],strength)
                self.assertEqual(reader.augmentation_config['horizontal_flip_probability'],.5)
                results[profile], label = reader[0]
                repeated = LocalImageDataset('train',root,run,augment=True,normalize=False,augmentation_profile=profile)
                np.testing.assert_array_equal(results[profile],repeated[0][0])
                self.assertEqual(label,0)
                self.assertEqual(results[profile].shape,(3,4,4))
                self.assertTrue(((results[profile]>=0)&(results[profile]<=1)).all())
            self.assertFalse(np.array_equal(results['suave'],results['fuerte']))
            self.assertEqual((run/'image.png').read_bytes(),original)

    def test_both_profiles_are_prohibited_outside_train(self):
        for split in ['validation','test']:
            for profile in AUGMENTATION_PROFILES:
                with self.assertRaisesRegex(ValueError,'solo se aplican'):
                    LocalImageDataset(split,augment=True,augmentation_profile=profile)
        with self.assertRaisesRegex(ValueError,'desconocido'):
            LocalImageDataset('train',augmentation_profile='otro')
