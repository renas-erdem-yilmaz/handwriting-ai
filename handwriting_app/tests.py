"""Local temporary model mocks; PDF bytes and actual Tk worker behavior are tested."""
import io
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
import numpy as np
from unittest.mock import patch
from PIL import Image,ImageDraw
from pypdf import PdfReader
from page_engine.common import ROOT
from .service import Settings,Runner,Cancelled,parse_document,export_pdf,destination,friendly_error
from .gui import Application

class MockPipeline:
    identity={'synthetic_unit_test_only':True}
    style_config={'ink_opacity_target':.82}
    def __init__(self,adapter=None,device='auto'):self.device_option=device

class MockStore:
    processing={'mode':'adaptive'}
    def __init__(self,pipeline,settings,initial_count=1):self.pipeline=pipeline;self.settings=settings;self.hits=0;self.generated=0
    def obtain(self,word,seed):
        self.hits+=1;image=Image.new('L',(max(14,len(word)*12),35),255)
        ImageDraw.Draw(image).rectangle((2,7,image.width-3,29),fill=60)
        return image,{'id':'fixture','bundle_id':'fixture','candidate_index':0,'flags':[],'status':'pending'}

class Checks(unittest.TestCase):
    def setUp(self):
        (ROOT/'outputs/app_tests').mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/app_tests');self.dir=Path(self.temp.name)
        self.runner=Runner(MockPipeline,MockStore)
        from page_engine.math_symbols.renderer import render
        def empty_library(*args,**kwargs):
            kwargs['library']=self.dir/'EMPTY_UNIT_TEST_SYMBOL_LIBRARY'
            return render(*args,**kwargs)
        self.patches=[patch('handwriting_app.service.render_math',side_effect=empty_library),
                      patch('page_engine.layout.render_math',side_effect=empty_library)]
        for item in self.patches:item.start()
    def tearDown(self):
        for item in self.patches:item.stop()
        self.temp.cleanup()
    def generate(self,text,**kw):return self.runner.generate(text,Settings(**kw),self.dir)

    def test_single_page_and_no_symbols_required(self):
        result=self.generate('# Notes\n\nA force acts.\n\n$a = F/m$')
        self.assertEqual(len(result['pages']),1)
        self.assertTrue(result['app']['input_preservation_verified'])
        self.assertEqual(result['cache_hits'],4)
        self.assertEqual(result['positions'][-1]['rendering']['source'],'typeset_math_not_handwriting')
        self.assertIn('fraction_bar',result['positions'][-1]['rendering']['missing_symbols'])

    def test_long_text_paragraphs_multipage_letter_lined(self):
        text='\n\n'.join([' '.join(['handwriting']*60)]*8)
        result=self.generate(text,page='Letter',background='lined')
        self.assertGreater(len(result['pages']),1)
        self.assertEqual(len(result['word_records']),480)
        self.assertEqual(len(result['document']['blocks']),8)
        self.assertEqual([r['text'] for r in result['word_records']],['handwriting']*480)
        with Image.open(ROOT/result['pages'][0]) as image:
            self.assertEqual(image.size,(1275,1650))
            self.assertGreater(int(np.all(np.asarray(image)==(218,229,240),axis=2).sum()),100)

    def test_empty_invalid_and_malformed_equation(self):
        for text in ['', '   ','# ','bad $equation','bad 123','hello 😀','$$ $$','$a+$']:
            with self.assertRaises(ValueError,msg=text):parse_document(text,Settings())
        for settings in [Settings(page='bad'),Settings(x_height=8),Settings(margins=(0,16,16,16)),Settings(line_spacing=3),Settings(x_height=float('nan'))]:
            with self.assertRaises(ValueError):parse_document('hello',settings)

    def test_blank_lines_and_mixed_math_input_preserved(self):
        doc=parse_document('# Notes\n\nSpeed $v_0+at$ changes.\n\n$$x = 1/2 at^2$$',Settings())
        self.assertEqual([b['kind'] for b in doc['blocks']],['heading','paragraph','math'])
        self.assertEqual(doc['blocks'][1]['text'],'Speed $v_0+at$ changes.')
        self.assertEqual(doc['blocks'][2]['text'],'x = 1/2 at^2')

    def test_pdf_exact_pixel_page_order_count_and_size(self):
        result=self.generate('\n\n'.join([' '.join(['test']*50)]*12),page='Letter')
        path=export_pdf(result,self.dir/'fixture.pdf');reader=PdfReader(path)
        self.assertEqual(len(reader.pages),len(result['pages']));self.assertGreater(len(reader.pages),1)
        for page,image_path in zip(reader.pages,result['pages']):
            self.assertAlmostEqual(float(page.mediabox.width),612,places=4);self.assertAlmostEqual(float(page.mediabox.height),792,places=4)
            image_objects=[v.get_object() for v in page['/Resources']['/XObject'].values() if v.get_object()['/Subtype']=='/Image']
            self.assertEqual(len(image_objects),1)
            with Image.open(ROOT/image_path) as image:self.assertEqual(image_objects[0].get_data(),image.convert('RGB').tobytes())
        with self.assertRaises(FileExistsError):export_pdf(result,path)

    def test_protected_destinations_and_changed_preview(self):
        for path in ['data/pdfs/new.pdf','reforge/new.pdf','training/new.pdf','.reforge-cache/new.pdf']:
            with self.assertRaises(ValueError):destination(path)
        result=self.generate('test')
        with Image.open(ROOT/result['pages'][0]) as image:
            changed=image.copy();changed.putpixel((0,0),(0,0,0));changed.save(ROOT/result['pages'][0])
        with self.assertRaises(ValueError):export_pdf(result,self.dir/'changed.pdf')

    def test_missing_models_recovery_and_cancel(self):
        def missing(*args):raise FileNotFoundError('missing pretrained checkpoint')
        runner=Runner(missing,MockStore)
        with self.assertRaises(FileNotFoundError):runner.generate('test',Settings(),self.dir)
        runner.pipeline_factory=MockPipeline
        self.assertEqual(len(runner.generate('test',Settings(),self.dir)['pages']),1)
        event=threading.Event()
        def progress(done,total,message):
            if message.startswith('Prepared'):event.set()
        with self.assertRaises(Cancelled):self.runner.generate('test again',Settings(),self.dir,progress,event)
        self.assertIn('cpu',friendly_error(RuntimeError('CUDA out of memory')))

    def test_gui_worker_responsiveness_preview_navigation_and_dirty_export(self):
        class SlowRunner:
            def generate(inner,*args):
                time.sleep(.15)
                return self.generate(' '.join(['test']*300),line_spacing=18)
        root=tk.Tk();root.withdraw()
        try:
            app=Application(root,SlowRunner());app.generate()
            self.assertTrue(app.busy)
            heartbeat=[];root.after(20,lambda:heartbeat.append(True))
            deadline=time.monotonic()+15
            while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertFalse(app.busy);self.assertTrue(heartbeat);self.assertIsNotNone(app.manifest)
            self.assertEqual(str(app.export_button['state']),'normal')
            app.navigate(1);self.assertEqual(app.page_index,1)
            app.zoom.set('100%');app.show_page();self.assertEqual(app.photo.width(),1240)
            pdf_path=self.dir/'gui_export.pdf'
            with patch('handwriting_app.gui.filedialog.asksaveasfilename',return_value=str(pdf_path)):app.export()
            deadline=time.monotonic()+8
            while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertFalse(app.busy)
            self.assertEqual(len(PdfReader(pdf_path).pages),len(app.manifest['pages']))
            app.vars['x_height'].set('3.5');self.assertTrue(app.dirty)
            self.assertEqual(str(app.export_button['state']),'disabled')
            app.close()
        finally:
            if not app.closed:
                app.closed=True;root.after_cancel(app.poll_id);root.destroy()

if __name__=='__main__':unittest.main()
