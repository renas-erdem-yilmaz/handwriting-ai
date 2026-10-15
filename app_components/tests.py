"""Streamlit and review integration tests; synthetic fixtures never enter user caches."""
import io
import contextlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest
from page_engine.common import ROOT
from page_engine.candidates import CandidateStore
from handwriting_app.service import Runner,Settings
from .backend import Backend,style_profile
from .resources import get_backend

def wait(job,timeout=20):
    deadline=time.monotonic()+timeout
    while not job.state()[0] and time.monotonic()<deadline:time.sleep(.03)
    done,result,error,_=job.state()
    if not done:raise AssertionError('Background job timed out')
    if error:raise AssertionError(error)
    return result

class Checks(unittest.TestCase):
    def setUp(self):
        self.silence=contextlib.redirect_stdout(io.StringIO());self.silence.__enter__()
        parent=ROOT/'outputs/app_tests/streamlit_v1';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.dir=Path(self.temp.name);self.calls=[];self.constructed=[]
        test=self
        class Model:
            style_config={'ink_opacity_target':.82}
            identity={'synthetic_unit_test_fixture':True}
            def __init__(inner,adapter=None,device='auto',style_config=None):
                inner.device_option=device;test.constructed.append(inner)
            def word(inner,text,seed,steps,guidance):
                test.calls.append((text,seed));image=Image.new('L',(max(50,len(text)*14+20),45),255)
                ImageDraw.Draw(image).text((10,7),text,fill=35,font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22))
                return image
        self.Model=Model
        def store_factory(pipeline,settings,initial_count=1):
            store=CandidateStore(pipeline,settings,directory=self.dir/'reviews',initial_count=initial_count)
            store.raw.directory=self.dir/'raw';store.raw.directory.mkdir(exist_ok=True)
            return store
        self.backend=Backend(Runner(Model,store_factory),str(self.dir/'runs'))
        get_backend.clear()
    def tearDown(self):
        get_backend.clear();self.temp.cleanup()
        self.silence.__exit__(None,None,None)

    def test_a4_equations_fallback_and_lossless_exports(self):
        snapshot=wait(self.backend.generate('# Notes\n\nA force acts.\n\n$a = F/m$\n\n$θ = α + β$',Settings()))
        self.assertEqual(len(snapshot['png']),1)
        self.assertEqual(len(PdfReader(io.BytesIO(snapshot['pdf'])).pages),1)
        self.assertEqual(snapshot['manifest']['positions'][-1]['rendering']['source'],'typeset_math_not_handwriting')
        for data,path in zip(snapshot['png'],snapshot['manifest']['pages']):
            with Image.open(io.BytesIO(data)) as exported,Image.open(ROOT/path) as preview:
                self.assertEqual(exported.tobytes(),preview.tobytes());self.assertAlmostEqual(exported.info['dpi'][0],150,delta=.02)

    def test_notebook_multiline_multipage(self):
        text='\n\n'.join([' '.join(['handwriting']*50)]*4)
        snapshot=wait(self.backend.generate(text,Settings(page='Notebook',background='lined')))
        self.assertGreater(len(snapshot['png']),1)
        self.assertEqual(len(snapshot['manifest']['word_records']),200)
        reader=PdfReader(io.BytesIO(snapshot['pdf']))
        self.assertEqual(len(reader.pages),len(snapshot['png']))
        self.assertAlmostEqual(float(reader.pages[0].mediabox.width),176*72/25.4,places=4)

    def test_review_regeneration_persistence_and_cache_only_refresh(self):
        snapshot=wait(self.backend.generate('A force acts.',Settings(candidates=2)))
        count=len(self.calls);constructed=len(self.constructed)
        changed=wait(self.backend.choose(snapshot,1,1,'approved','Synthetic-fixture review test'))
        self.assertEqual(len(self.calls),count);self.assertEqual(len(self.constructed),constructed)
        self.assertEqual(self.backend.bundle(changed,1)['selected'],1)
        self.assertEqual(self.backend.bundle(changed,1)['status'],'approved')
        before_other=[r['cache_id'] for r in changed['manifest']['word_records'] if r['index']!=1]
        added=wait(self.backend.regenerate(changed,1))
        self.assertEqual(len(self.calls),count+1);self.assertEqual(added['bundle']['selected'],1)
        final=wait(self.backend.choose(changed,1,2,'pending','Test selection, not human-approved'))
        self.assertEqual(len(self.calls),count+1)
        self.assertEqual([r['cache_id'] for r in final['manifest']['word_records'] if r['index']!=1],before_other)
        again=wait(self.backend.refresh(final));self.assertEqual(len(self.calls),count+1)
        self.assertEqual(again['manifest']['new_words'],0)

    def test_invalid_input_missing_model_recovery(self):
        job=self.backend.generate('bad 123',Settings())
        deadline=time.monotonic()+10
        while not job.state()[0] and time.monotonic()<deadline:time.sleep(.01)
        self.assertIn('Unsupported prose token',job.state()[2]);self.assertFalse(self.calls)
        original=self.backend.runner.pipeline_factory
        def missing(*args,**kwargs):raise FileNotFoundError(2,'missing',str(self.dir/'missing_checkpoint.pt'))
        self.backend.runner.pipeline_factory=missing;job=self.backend.generate('test',Settings())
        while not job.state()[0] and time.monotonic()<deadline:time.sleep(.01)
        self.assertIn('Restore',job.state()[2]);self.assertNotIn(str(self.dir),job.state()[2])
        self.backend.runner.pipeline_factory=original
        self.assertEqual(len(wait(self.backend.generate('test',Settings()))['png']),1)

    def test_style_selection_and_settings(self):
        from page_engine.common import read_json
        refs=read_json(ROOT/'page_engine/styles/erdem.json')['references']
        self.assertEqual(style_profile(refs),('page_engine/styles/erdem.json',None))
        with self.assertRaises(ValueError):style_profile(refs[:4])
        with self.assertRaises(ValueError):Settings(natural_variation=.1).validate()
        settings=Settings(page='Notebook',opacity_target=.9,natural_variation=.05)
        settings.validate()
        from handwriting_app.service import parse_document
        scaled=parse_document('$F = ma$',Settings(x_height=4))
        self.assertAlmostEqual(scaled['layout']['math_font_points'],15*4/3.1)

    def ui_wait(self,app,timeout=20):
        deadline=time.monotonic()+timeout
        while app.session_state['job'] is not None and time.monotonic()<deadline:
            time.sleep(.03);app.run(timeout=20)
        self.assertIsNone(app.session_state['job'])
        self.assertEqual(len(app.exception),0,[e.message for e in app.exception])

    def test_streamlit_start_generate_reruns_review_and_regenerate(self):
        with patch('app_components.resources.Backend',return_value=self.backend):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
            self.assertEqual(len(app.exception),0);self.assertFalse(self.constructed);self.assertFalse(self.calls)
            app.text_area(key='editor_text').set_value('A force acts.\n\n$F = ma$')
            app.selectbox(key='candidate_count').set_value(2)
            app.button(key='generate').click().run();self.ui_wait(app)
            snapshot=app.session_state['snapshot'];self.assertIsNotNone(snapshot)
            count=len(self.calls)
            app.select_slider(key='preview_zoom').set_value('Enlarge').run()
            app.slider(key='line_gap').set_value(9.).run()
            app.run();self.assertEqual(len(self.calls),count)
            self.assertEqual(app.session_state['snapshot']['pdf'],snapshot['pdf'])
            bundle=self.backend.bundle(snapshot,0)
            app.radio(key='pick_'+bundle['id']).set_value(1).run()
            app.button(key='approve_word').click().run();self.ui_wait(app)
            self.assertEqual(self.backend.bundle(app.session_state['snapshot'],0)['status'],'approved')
            self.assertEqual(len(self.calls),count)
            app.button(key='regenerate_word').click().run();self.ui_wait(app)
            self.assertEqual(len(self.calls),count+1)
            self.assertEqual(len(app.exception),0)
            self.assertGreater(len(app.get('download_button')),1)

if __name__=='__main__':unittest.main()
