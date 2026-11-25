import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import Request, urlopen
from PIL import Image

spec = importlib.util.spec_from_file_location('review_gallery', Path(__file__).resolve().parents[1]/'tools/revisar_fotos.py')
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        run = self.root/'data/processed/runs/test'
        run.mkdir(parents=True)
        (self.root/'config').mkdir()
        self.manual = self.root/'config/training_visual_review.csv'
        self.manual.write_text(','.join(gallery.COLUMNS)+'\n',encoding='utf-8')
        (self.root/'data/processed/CURRENT.json').write_text(json.dumps({'run_dir':'runs/test'}))
        path = self.root/'data/processed/image.jpg'
        Image.new('RGB',(4,4),(100,50,70)).save(path)
        rows = [{'article_id':f'{i:010d}', 'file_sha256':f'original-{i}', 'prepared_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'assigned_split':'train','split':'train','index_group_name':'Ladieswear' if i<13 else 'Baby/Children',
                 'image_path':'image.jpg','label':'Shirt','auto_visual_decision':'uncertain'} for i in range(14)]
        with gzip.open(run/'image_manifest.csv.gz','wt',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (run/'checksums.json').write_text(json.dumps({'image_manifest.csv.gz':hashlib.sha256((run/'image_manifest.csv.gz').read_bytes()).hexdigest()}))

    def tearDown(self):
        self.temp.cleanup()

    def test_autosave_resume_change_and_pending_only(self):
        store=gallery.ReviewStore(self.root)
        active_before=self.manual.read_bytes()
        batch=store.batch()
        self.assertEqual(len(batch['cards']),12)
        self.assertTrue(batch['more'])
        store.save('0000000000','exclude_detail')
        resumed=gallery.ReviewStore(self.root)
        self.assertNotIn('0000000000',[x['article_id'] for x in resumed.batch()['cards']])
        resumed.save('0000000000','uncertain','No se reconoce bien')
        self.assertEqual(resumed.batch(mode='uncertain')['cards'][0]['reason'],'No se reconoce bien')
        resumed.save('0000000000','keep')
        self.assertEqual(gallery.ReviewStore(self.root).batch(mode='all')['cards'][0]['decision'],'keep')
        self.assertEqual(self.manual.read_bytes(),active_before)
        self.assertEqual(len(resumed.rows),13)
        with self.assertRaises(ValueError):resumed.save('0000000013','keep')
        with self.assertRaises(ValueError):resumed.save('0000000000','borrar')

    def test_page_image_and_button_endpoint_with_synthetic_data(self):
        server,url=gallery.start_review(self.root,open_browser=False)
        try:
            with urlopen(url,timeout=3) as response:
                page=response.read().decode('utf-8')
            self.assertIn('Cada clic se guarda automáticamente',page)
            self.assertIn("value=\"uncertain\"",page)
            with urlopen(url+'image/0000000000',timeout=3) as response:
                self.assertEqual(response.headers['Content-Type'],'image/jpeg')
            request=Request(url+'decision',data=json.dumps({'article_id':'0000000000','decision':'exclude_detail'}).encode(),
                            headers={'Content-Type':'application/json'},method='POST')
            with urlopen(request,timeout=3) as response:self.assertTrue(json.load(response)['saved'])
            self.assertEqual(gallery.ReviewStore(self.root).pending['0000000000']['decision'],'exclude_detail')
        finally:
            server.shutdown()
            server.server_close()

    def test_default_excludes_clip_keep_and_already_discarded_photos(self):
        store=gallery.ReviewStore(self.root)
        store.rows['0000000000']['auto_visual_decision']='keep'
        store.rows['0000000001']['split']='excluded'
        ids=[row['article_id'] for row in store.batch()['cards']]
        self.assertEqual(len(ids),11)
        self.assertNotIn('0000000000',ids)
        self.assertNotIn('0000000001',ids)
        self.assertEqual(store.batch(mode='pending')['total'],13)
