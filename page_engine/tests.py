"""Geometry/math/cache checks; fixture images are only unit-test fixtures."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from .model import prepare_imports
from .normalize import normalize, ink_box, assess
from .layout import tokens, build
from .math_render import render
from .common import ROOT
from .words import WordStore

class Checks(unittest.TestCase):
    def test_bounded_retry_and_cache(self):
        class Model:
            identity = {'fixture': True}
            calls = 0
            def word(self, text, seed, steps, guidance):
                self.calls += 1
                image = Image.new('L', (120, 40), 255)
                if self.calls > 1:
                    from PIL import ImageFont
                    ImageDraw.Draw(image).text((10, 5), 'test', fill=0, font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24))
                return image
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs/layout_tests') as directory:
            model = Model(); store = WordStore(model, {'steps':20, 'guidance':3.0, 'retries':1, 'ocr':False})
            store.directory = Path(directory)
            image, record = store.obtain('test', 1)
            self.assertEqual(model.calls, 2)
            self.assertEqual(record['attempts'][0]['flags'], ['blank'])
            self.assertEqual(record['selected_attempt'], 1)
            image, cached = store.obtain('test', 1)
            self.assertEqual(model.calls, 2)
            self.assertEqual(store.hits, 1)
            self.assertFalse(cached['spelling_verified'])
    def test_math_and_markup(self):
        for expression in [r'α,β,θ,φ,Δ,ω', r'a=\frac{F}{m}⇒v_0+at^2', r'x\leq y\neq0', r'\sqrt{x}+\sum_{i=0}^{2}i']:
            image, baseline, meta = render(expression)
            self.assertGreater(image.width, 10)
            self.assertTrue(0 <= baseline < image.height)
            self.assertEqual(meta['source'], 'typeset_math_not_handwriting')
        with self.assertRaises(ValueError): render(r'\madeUpCommand{x}')
        self.assertEqual(tokens('speed $v_0+at$ changes.'), [('word', 'speed'), ('math', 'v_0+at'), ('word', 'changes.')])
        with self.assertRaises(ValueError): tokens('bad $math')

    def test_normalization(self):
        prepare_imports()
        fixture = Image.new('L', (300, 120), 255)
        ImageDraw.Draw(fixture).rectangle((40, 30, 239, 69), fill=0)
        self.assertEqual(ink_box(fixture), (40, 30, 240, 70))
        image, baseline, _ = normalize(fixture, 'test', 28)
        self.assertEqual(image.size, (140, 28))
        self.assertTrue(0 <= baseline < 28)
        self.assertEqual(assess(Image.new('L', (20,20), 255), 'blank')[0], ['blank'])

    def test_actual_width_wrapping_and_pagination(self):
        prepare_imports()
        class Store:
            class Model: identity = {'fixture': True}
            pipeline = Model(); settings = {}; hits = generated = 0
            def obtain(self, text, seed):
                image = Image.new('L', (80, 25), 255)
                ImageDraw.Draw(image).rectangle((1, 1, 78, 23), fill=0)
                return image, {'id': 'fixture', 'status': 'approved', 'flags': []}
        profile = {'margins_fraction': [.1,.1,.1,.1], 'line_pitch_fraction': .08, 'paragraph_extra_lines': .5, 'heading_align': 'center'}
        document = {'blocks': [{'kind': 'paragraph', 'text': ' '.join(['test'] * 120)}],
                    'layout': {'size_mm': [80, 100], 'line_spacing_mm': 8, 'margins_mm': [8,8,8,8]}}
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs/layout_tests') as directory:
            result = build(document, Store(), str(Path(directory) / 'fixture'), profile)
            self.assertGreater(len(result['pages']), 1)
            self.assertEqual(len(result['positions']), 120)
            left, top, right, bottom = result['geometry']['usable_box']
            for pos in result['positions']:
                x,y,r,b = pos['box']
                self.assertTrue(left <= x < r <= right and top <= y < b <= bottom)
            grouped = {}
            for pos in result['positions']:
                key = (pos['page'],pos['block'],pos['line'])
                for other in grouped.get(key, []): self.assertGreaterEqual(pos['box'][0], other['box'][2])
                grouped.setdefault(key, []).append(pos)
            with self.assertRaises(FileExistsError): build(document, Store(), str(Path(directory) / 'fixture'), profile)

if __name__ == '__main__': unittest.main()
