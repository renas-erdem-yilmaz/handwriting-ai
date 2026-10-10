"""Ink/x-height tests with independently constructed glyph geometry."""
import unittest
from PIL import Image,ImageDraw
from .writing_scale import measure,calibrate
from .consistency import normalize_adaptive

def fixture(body=20,ascender=False,descender=False):
    image=Image.new('RGBA',(260,100),(0,0,0,0));draw=ImageDraw.Draw(image)
    top=35;baseline=top+body-1
    for x in [50,85,120,155]:draw.rectangle((x,top,x+18,baseline),fill=(30,30,30,255))
    if ascender:draw.rectangle((50,top-body//2,54,baseline),fill=(30,30,30,255))
    if descender:draw.rectangle((172,top,177,baseline+body//2),fill=(30,30,30,255))
    draw.point((248,4),fill=(0,0,0,255)) # Remote background noise.
    return image

class Checks(unittest.TestCase):
    def test_alpha_noise_and_scale_invariance(self):
        for source_body in [12,20,35]:
            image=fixture(source_body)
            m=measure(image,'moon')
            self.assertTrue(m['confidence'])
            self.assertEqual(m['x_height'],source_body)
            self.assertGreater(m['box'][0],0);self.assertGreater(m['box'][1],4)
            result,base,meta=normalize_adaptive(image,'moon',28,x_height=20)
            self.assertEqual(result.height,20)
            self.assertAlmostEqual(meta['estimated_x_height_rendered'],20,delta=.5)
    def test_ascender_descender_preservation(self):
        for asc,desc,text in [(True,False,'home'),(False,True,'young'),(True,True,'broomy')]:
            image=fixture(20,asc,desc);m=measure(image,text)
            self.assertEqual(m['x_height'],20)
            normalized,baseline,meta=normalize_adaptive(image,text,28,x_height=20)
            self.assertGreater(normalized.height,20)
            if desc:self.assertGreater(normalized.height-baseline-1,4)
            if asc:self.assertGreater(baseline,24)
            box=m['box'];self.assertAlmostEqual(normalized.width/normalized.height,(box[2]-box[0])/(box[3]-box[1]),delta=.04)
    def test_fallback_and_variation(self):
        m=measure(fixture(20,True),'b')
        self.assertFalse(m['confidence'])
        profile=calibrate([measure(fixture(20,True),'home') for _ in range(4)])
        self.assertEqual(profile['reliable_word_counts']['ascenders'],4)
        _,_,meta=normalize_adaptive(fixture(20,True),'b',28,variation=1.50,x_height=20,calibration=profile,measurement=m)
        self.assertEqual(meta['size_variation_factor'],1.05)
        self.assertIn('fallback',meta['baseline_method'])
        self.assertAlmostEqual(meta['estimated_x_height_rendered'],21,delta=.7)

if __name__=='__main__':unittest.main()
