import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from src.data.zoom_geometry import border_foreground, geometry_scores


class ZoomGeometryTests(unittest.TestCase):
    def test_isolated_dark_garment_and_edge_contact_with_padding(self):
        whole = Image.new('RGB',(64,64),'white')
        whole.paste('black',(16,8,48,56))
        self.assertEqual(border_foreground(whole,64,64),0)
        detail = Image.new('RGB',(64,64),'black')
        self.assertEqual(border_foreground(detail,64,64),1)
        padded = Image.new('RGB',(64,64),'white')
        padded.paste('black',(16,0,48,64))
        # El relleno blanco de una imagen estrecha no debe ocultar sus bordes.
        self.assertEqual(border_foreground(padded,32,64),1)

    def test_cache_reuses_measure_without_reopening_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/'data/processed';folder.mkdir(parents=True)
            path=folder/'photo.png'
            Image.new('RGB',(64,64),'black').save(path)
            rows=pd.DataFrame([{'article_id':'001','prepared_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'image_path':'photo.png','width':64,'height':64}])
            first=geometry_scores(rows,root)
            with patch('src.data.zoom_geometry.Image.open',side_effect=AssertionError('No releer fotos')):
                second=geometry_scores(rows,root)
            np.testing.assert_array_equal(first,second)
            rows['prepared_sha256']='incorrecto'
            with self.assertRaisesRegex(ValueError,'modificada'):
                geometry_scores(rows,root)
