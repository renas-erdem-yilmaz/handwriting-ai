"""Synthetic temporary fixtures ONLY; never creates approved user symbols."""
import tempfile
import unittest
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageDraw
from .library import Library, prepared
from .syntax import parse, tex, required, SyntaxError
from .renderer import render
from .collector import Collector
from .__main__ import EXAMPLES
from ..common import ROOT, sha

class Checks(unittest.TestCase):
    def setUp(self):
        (ROOT/'outputs/math_tests').mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/math_tests')
        self.directory=Path(self.temp.name);self.lib=Library(self.directory/'fixture_library')
        self.source=self.directory/'SYNTHETIC_UNIT_TEST_SOURCE.png'
        image=Image.new('RGB',(60,65),'white');d=ImageDraw.Draw(image)
        d.line((15,8,42,46),fill='black',width=3);d.line((12,45,42,8),fill='black',width=3)
        image.save(self.source)
    def tearDown(self):self.temp.cleanup()

    def add(self,label):
        e=self.lib.capture(self.source,(0,0,60,65),label,baseline=.85,height_em=.10 if label=='fraction_bar' else 1)
        return self.lib.update(e['id'],label,220,.85,e['height_em'],approve=True,reviewed=True)

    def test_approval_gate_and_source_preservation(self):
        digest=sha(self.source);entry=self.lib.capture(self.source,(0,0,60,65),'x')
        self.assertEqual(self.lib.variants('x'),[])
        with self.assertRaises(ValueError):self.lib.update(entry['id'],'x',220,.85,1,approve=True)
        self.lib.update(entry['id'],'x',220,.85,1,approve=True,reviewed=True)
        self.assertEqual(len(Library(self.lib.directory).variants('x')),1)
        self.lib.update(entry['id'],'y',220,.85,1)
        self.assertEqual(self.lib.variants('x'),[]);self.assertEqual(self.lib.variants('y'),[])
        self.assertEqual(sha(self.source),digest)

    def test_restricted_syntax_and_precedence(self):
        self.assertEqual(required(parse('x = 1/2 at^2')),set('x=12at')|{'fraction_bar'})
        self.assertIn(r'\frac{1 }{2 }',tex(parse('1/2 at^2')))
        self.assertIn('(',tex(parse('{a+b}c')))
        self.assertEqual(required(parse('v_0^2')),set('v02'))
        division=parse('a÷b/c')
        self.assertEqual(division.kind,'fraction')
        self.assertEqual(division.children[0].kind,'row')
        self.assertIn(r'\div',tex(division.children[0]))
        for malformed in ['a+','a/','v^','(a+b]','v_0_1','a^^2','()']:
            with self.assertRaises(SyntaxError,msg=malformed):parse(malformed)

    def test_all_examples_missing_symbols_fall_back_exactly(self):
        for expression in EXAMPLES:
            image,baseline,meta=render(expression,library=self.lib)
            self.assertEqual(meta['source'],'typeset_math_not_handwriting')
            self.assertEqual(meta['input_expression'],expression)
            self.assertEqual(set(meta['missing_symbols']),required(parse(expression)))
            self.assertTrue(0<=baseline<image.height)
            self.assertGreater(image.width,10)
        _,_,meta=render(r'\sqrt{x}',library=self.lib)
        self.assertEqual(meta['fallback_reason'],'outside_restricted_syntax')

    def test_assembled_fractions_scripts_variants_and_bounds(self):
        expressions=EXAMPLES+['(a+b)/(c-d)','v_0^{2+1} ⇒ θ','α + β × Δ / ω','φ_1^2']
        for label in set().union(*(required(parse(e)) for e in expressions)):self.add(label)
        self.add('v')  # Multiple approved variants.
        for expression in expressions:
            image,baseline,meta=render(expression,library=self.lib,x_height=20,seed=5)
            self.assertEqual(meta['source'],'handwritten_math_symbols');self.assertFalse(meta['missing_symbols'])
            self.assertTrue(0<=baseline<image.height)
            for glyph in meta['glyphs']:
                x,y,r,b=glyph['box'];self.assertTrue(0<=x<r<=image.width and 0<=y<b<=image.height)
                self.assertTrue(.975<=glyph['scale_variation']<=1.025)
            again,again_base,again_meta=render(expression,library=self.lib,x_height=20,seed=5)
            self.assertEqual(image.tobytes(),again.tobytes());self.assertEqual(baseline,again_base)
        _,_,meta=render('a/b',library=self.lib)
        self.assertTrue(any(g.get('fraction_bar_extended') for g in meta['glyphs']))

    def test_changed_approved_crop_is_not_silently_used(self):
        entry=self.add('x');Image.new('RGB',(5,5),'black').save(self.lib.directory/entry['image'])
        with self.assertRaises(ValueError):render('x',library=self.lib)

    def test_handwritten_page_integration_and_pagination(self):
        from ..layout import build
        for label in set().union(*(required(parse(e)) for e in EXAMPLES)):self.add(label)
        class NoProse:
            class Pipeline:identity={'synthetic_unit_test':True}
            pipeline=Pipeline();settings={};hits=generated=0
            def obtain(self,*args):raise AssertionError('No text inference expected')
        profile={'margins_fraction':[.1,.1,.1,.1],'line_pitch_fraction':.08,'paragraph_extra_lines':.5,'heading_align':'center'}
        document={'blocks':[{'kind':'math','text':e} for e in EXAMPLES*2],
                  'math':{'library':str(self.lib.directory),'variation':0},
                  'layout':{'size_mm':[90,100],'margins_mm':[8,8,8,8],'line_spacing_mm':8,'x_height_mm':3}}
        result=build(document,NoProse(),str(self.directory/'SYNTHETIC_TEST_PAGE'),profile)
        self.assertGreater(len(result['pages']),1)
        for pos in result['positions']:
            self.assertEqual(pos['rendering']['source'],'handwritten_math_symbols')
            self.assertAlmostEqual(pos['rendering']['target_x_height_pixels'],3*150/25.4)
            x,y,r,b=pos['box'];left,top,right,bottom=result['geometry']['usable_box']
            self.assertTrue(left<=x<r<=right and top<=y<b<=bottom)

    def test_transparency_and_blank_detection(self):
        with self.assertRaises(ValueError):prepared(Image.new('RGBA',(10,10),(0,0,0,0)))
        with self.assertRaises(ValueError):self.lib.capture(self.source,(-1,0,20,20),'x')

    def test_tk_crop_save_review_resume(self):
        root=tk.Tk();root.withdraw()
        try:
            app=Collector(root,self.lib);app.open_page(self.source);app.raw=Image.open(self.source).copy();app.rect=(0,0,60,65)
            app.label.set('x');app.save(False)
            self.assertEqual(len(self.lib.data['entries']),1);self.assertEqual(self.lib.variants('x'),[])
            app.reviewed.set(True);app.save(True)
            self.assertEqual(len(Library(self.lib.directory).variants('x')),1)
            app.entries.selection_set(0);app.select_entry();self.assertEqual(app.label.get(),'x');self.assertFalse(app.reviewed.get())
            app.reset();self.assertEqual(app.label.get(),'')
            root.update()
        finally:root.destroy()

if __name__=='__main__':unittest.main()
