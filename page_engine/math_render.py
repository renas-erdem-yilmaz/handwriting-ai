"""Standard-font math; never presented as generated handwriting."""
import io
import warnings
import numpy as np
from PIL import Image
from .normalize import ink_box

SYMBOLS = {'α':r'\alpha', 'β':r'\beta', 'θ':r'\theta', 'φ':r'\phi', 'Δ':r'\Delta', 'ω':r'\omega',
           'γ':r'\gamma', 'π':r'\pi', 'μ':r'\mu', 'σ':r'\sigma', 'λ':r'\lambda',
           '⇒':r'\Rightarrow', '→':r'\rightarrow', '≤':r'\leq', '≥':r'\geq', '≠':r'\neq',
           '±':r'\pm', '×':r'\times', '÷':r'\div', '∞':r'\infty', '∑':r'\sum', '∫':r'\int',
           '²':'^{2}', '³':'^{3}', '₀':'_{0}', '₁':'_{1}', '₂':'_{2}'}

def render(expression, dpi=150, font_size=16):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.mathtext import math_to_image, MathTextParser
    from matplotlib.font_manager import FontProperties
    for char, tex in SYMBOLS.items(): expression = expression.replace(char, tex + (' ' if tex.startswith('\\') else ''))
    if '$' in expression: raise ValueError('Math block should not contain dollar delimiters')
    formula = '$' + expression + '$'
    prop = FontProperties(size=font_size, math_fontfamily='stix')
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*Glyph.*missing.*')
        parsed = MathTextParser('agg').parse(formula, dpi=dpi, prop=prop)
        stream = io.BytesIO()
        math_to_image(formula, stream, prop=prop, dpi=dpi, format='png', color='black')
    original = Image.open(stream).convert('RGBA')
    background = Image.new('RGBA', original.size, 'white'); background.alpha_composite(original)
    image = background.convert('L')
    box = ink_box(image)
    # Parser metrics and math_to_image share the same glyph layout; account for tight cropping.
    baseline = max(0, min(box[3] - box[1] - 1, round(original.height - parsed.depth - box[1])))
    return image.crop(box), baseline, {'source': 'typeset_math_not_handwriting', 'expression': expression, 'font': 'STIX'}
