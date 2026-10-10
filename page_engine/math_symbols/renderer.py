"""Baseline-aware assembly; missing symbols trigger an honest whole-equation fallback."""
from dataclasses import dataclass
import random
from PIL import Image, ImageChops
from .library import Library, DEFAULT_LIBRARY
from .syntax import parse, required, tex, UnsupportedSyntax
from ..math_render import render as font_render

@dataclass
class Box:
    image: Image.Image
    baseline: int
    glyphs: list

def compose(items, baseline):
    """Items carry absolute local offsets; expand the canvas rather than clip scripts."""
    xmin=min(0,*(x for box,x,y in items));ymin=min(0,*(y for box,x,y in items))
    xmax=max(x+box.image.width for box,x,y in items);ymax=max(y+box.image.height for box,x,y in items)
    image=Image.new('L',(max(1,xmax-xmin),max(1,ymax-ymin)),255);glyphs=[]
    for box,x,y in items:
        x-=xmin;y-=ymin
        rect=(x,y,x+box.image.width,y+box.image.height)
        image.paste(ImageChops.darker(image.crop(rect),box.image),(x,y))
        for glyph in box.glyphs:
            g=dict(glyph);a,b,r,d=g['box'];g['box']=[a+x,b+y,r+x,d+y];glyphs.append(g)
    return Box(image,baseline-ymin,glyphs)

class Assembler:
    def __init__(self, library, seed, variation):
        self.library=library;self.rng=random.Random(seed);self.variation=variation

    def glyph(self,label,em):
        entry=self.rng.choice(self.library.variants(label))
        raw=self.library.image(entry)
        factor=1+self.rng.uniform(-self.variation,self.variation)
        height=max(2,round(em*entry['height_em']*factor))
        image=raw.resize((max(1,round(raw.width*height/raw.height)),height),Image.Resampling.LANCZOS)
        baseline=min(height-1,round(entry['baseline']*(height-1)))
        return Box(image,baseline,[{'label':label,'entry_id':entry['id'],'crop_sha256':entry['sha256'],
                'source_page':entry['source_page'],'source_sha256':entry['source_sha256'],
                'box':[0,0,image.width,image.height],'scale_variation':factor}])

    def assemble(self,node,em):
        if node.kind=='glyph':return self.glyph(node.value,em)
        if node.kind=='group':
            if node.value=='{}':return self.assemble(node.children[0],em)
            # Parentheses stay explicit; no attempt to infer absent grouping.
            from .syntax import Node
            return self.assemble(Node('row',children=(Node('glyph',node.value[0]),node.children[0],Node('glyph',node.value[1]))),em)
        if node.kind=='row':
            boxes=[self.assemble(child,em) for child in node.children]
            baseline=max(b.baseline for b in boxes);items=[];x=0
            for child,box in zip(node.children,boxes):
                operator=child.kind=='glyph' and child.value in '+−=×÷⇒→≤≥≠±'
                if operator and x:x+=round(em*.22)
                items.append((box,x,baseline-box.baseline));x+=box.image.width+max(1,round(em*(.22 if operator else .10)))
            return compose(items,baseline)
        if node.kind=='fraction':
            num=self.assemble(node.children[0],em*.88);den=self.assemble(node.children[1],em*.88)
            bar=self.glyph('fraction_bar',em)
            width=max(num.image.width,den.image.width)+max(4,round(em*.35))
            # Only fraction bars extend horizontally; glyph aspect ratios stay unchanged.
            bar.image=bar.image.resize((width,bar.image.height),Image.Resampling.LANCZOS)
            bar.glyphs[0].update(box=[0,0,width,bar.image.height],fraction_bar_extended=True)
            gap=max(2,round(em*.20));bar_y=num.image.height+gap
            items=[(num,(width-num.image.width)//2,0),(bar,0,bar_y),
                   (den,(width-den.image.width)//2,bar_y+bar.image.height+gap)]
            return compose(items,bar_y+bar.image.height//2+round(em*.4))
        if node.kind=='script':
            base,sup,sub=node.children;base=self.assemble(base,em);items=[(base,0,0)]
            x=base.image.width+max(1,round(em*.08))
            if sup is not None:
                box=self.assemble(sup,em*.64)
                items.append((box,x,base.baseline-round(em*.80)-box.baseline))
            if sub is not None:
                box=self.assemble(sub,em*.64)
                items.append((box,x,base.baseline+round(em*.45)-box.baseline))
            return compose(items,base.baseline)
        raise ValueError('Unknown node '+node.kind)

def render(expression,dpi=150,font_size=16,*,library=DEFAULT_LIBRARY,seed=42,x_height=None,variation=.025,mode='auto'):
    if mode not in ('auto','font'):raise ValueError('Math mode must be auto or font')
    if not 0<=variation<=.05:raise ValueError('Symbol variation must be within 0–5%')
    lib=library if isinstance(library,Library) else Library(library)
    info={'input_expression':expression,'symbol_library':lib.identity(),'seed':seed,'symbol_variation':variation}
    if mode=='font':
        image,baseline,meta=font_render(expression,dpi,font_size)
        meta.update(info,fallback_reason='font_mode_requested',missing_symbols=[])
        return image,baseline,meta
    try:node=parse(expression)
    except UnsupportedSyntax as error:
        # Preserve existing Mathtext functionality, including \sqrt and explicit LaTeX.
        image,baseline,meta=font_render(expression,dpi,font_size)
        meta.update(info,fallback_reason='outside_restricted_syntax',syntax_warning=str(error),missing_symbols=[])
        return image,baseline,meta
    needed=sorted(required(node));missing=[label for label in needed if not lib.variants(label)]
    info.update(required_symbols=needed,missing_symbols=missing,canonical_tex=tex(node))
    if missing:
        image,baseline,meta=font_render(tex(node),dpi,font_size)
        meta.update(info,fallback_reason='missing_approved_symbols')
        return image,baseline,meta
    em=float(x_height if x_height is not None else font_size*dpi/72*.55)
    if not 3<=em<=300:raise ValueError('Math x-height must be 3–300 pixels')
    box=Assembler(lib,seed,variation).assemble(node,em)
    return box.image,box.baseline,dict(info,source='handwritten_math_symbols',glyphs=box.glyphs,
                                     approved_symbol_crops=True,target_x_height_pixels=em)
