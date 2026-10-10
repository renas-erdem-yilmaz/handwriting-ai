"""Quality checks use explicitly synthetic fixtures, never delivered page images."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import ROOT,read_json,sha
from .candidates import CandidateStore
from .consistency import normalize_adaptive

class FixtureModel:
    identity={'fixture':'quality-tests'}
    style_config={'ink_opacity_target':.82}
    def __init__(self):self.calls=0
    def word(self,text,seed,steps,guidance):
        self.calls+=1
        image=Image.new('L',(160,64),255)
        ImageDraw.Draw(image).text((10,8),text,fill=20+seed%80,font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30))
        return image

class Checks(unittest.TestCase):
    def fixture(self,directory):
        model=FixtureModel();store=CandidateStore(model,{'steps':20,'guidance':3.,'ocr':False},directory=Path(directory)/'reviews')
        store.raw.directory=Path(directory)/'cache';store.raw.directory.mkdir()
        return model,store
    def test_candidates_decisions_resume_and_reuse(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/quality_tests') as directory:
            model,store=self.fixture(directory)
            bundle=store.ensure('test',100)
            self.assertEqual(model.calls,3)
            self.assertEqual(len(bundle['candidates']),3)
            self.assertEqual(len({c['seed'] for c in bundle['candidates']}),3)
            originals={c['image']:sha(ROOT/c['image']) for c in bundle['candidates']}
            store.decide(bundle['id'],1,'approved','Fixture UI test, not genuine handwriting approval')
            image,record=store.obtain('test',100)
            self.assertEqual(record['candidate_index'],1);self.assertEqual(record['status'],'approved')
            self.assertEqual(model.calls,3)
            store.append(bundle['id'],777)
            resumed=CandidateStore(model,store.settings,directory=store.directory)
            resumed.raw.directory=store.raw.directory
            self.assertEqual(resumed.load(bundle['id'])['selected'],1)
            self.assertEqual(resumed.load(bundle['id'])['status'],'approved')
            self.assertEqual(model.calls,4)
            resumed.reuse_approved=True
            reused=resumed.ensure('test',888)
            self.assertEqual(model.calls,4);self.assertEqual(reused['approval_inherited_from'],bundle['id'])
            self.assertEqual(reused['status'],'approved')
            resumed.decide(bundle['id'],status='problematic',note='Merged letters')
            self.assertEqual(resumed.load(bundle['id'])['problem_note'],'Merged letters')
            for path,digest in originals.items():self.assertEqual(sha(ROOT/path),digest)
            with self.assertRaises(ValueError):resumed.append(bundle['id'],777)
    def test_tonal_adjustment_and_geometry(self):
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',45)
        for color in [5,65,100]:
            image=Image.new('L',(400,100),255);ImageDraw.Draw(image).text((20,10),'body',font=font,fill=color)
            before=sha_image(image)
            normalized,base,meta=normalize_adaptive(image,'body',28)
            self.assertTrue(0<=base<normalized.height)
            self.assertAlmostEqual(meta['estimated_x_height_rendered'],28*.73,delta=1)
            box=meta['raw_ink_box'];ratio=(box[2]-box[0])/(box[3]-box[1])
            self.assertAlmostEqual(normalized.width/normalized.height,ratio,delta=.04)
            self.assertTrue(.82<=meta['ink_gain']<=1.18)
            self.assertEqual(sha_image(image),before)
    def test_tk_buttons_and_resume(self):
        import tkinter as tk
        from .review_ui import Reviewer
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/quality_tests') as directory:
            model,store=self.fixture(directory);bundle=store.ensure('test',100)
            session={'model':model.identity,'words':[{'index':0,'text':'test','seed':100,'bundle_id':bundle['id']}]}
            root=tk.Tk();root.withdraw()
            ui=Reviewer(root,session,store);root.update()
            self.assertEqual(len(ui.photos),3)
            ui.card_widgets[1][2].invoke();root.update()
            self.assertEqual(store.load(bundle['id'])['selected'],1)
            self.assertEqual(store.load(bundle['id'])['status'],'approved')
            ui.seed.set('1234');ui.regen.invoke()
            import time
            deadline=time.monotonic()+3
            while ui.busy and time.monotonic()<deadline:
                root.update();time.sleep(.01)
            self.assertFalse(ui.busy)
            self.assertEqual(len(store.load(bundle['id'])['candidates']),4)
            self.assertEqual(store.load(bundle['id'])['selected'],1)
            ui.note.set('Fixture malformed');ui.problem();root.update()
            self.assertEqual(store.load(bundle['id'])['status'],'problematic')
            root.destroy()
            root=tk.Tk();root.withdraw();resumed=Reviewer(root,session,store);root.update()
            self.assertIn('problematic',resumed.status.cget('text'))
            root.destroy()

def sha_image(image):
    import hashlib
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()

if __name__=='__main__':unittest.main()
