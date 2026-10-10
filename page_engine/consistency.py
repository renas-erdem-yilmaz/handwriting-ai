"""Conservative luminance and body-height normalization, never glyph repair."""
import numpy as np
from PIL import Image
from .writing_scale import measure,white_image

def normalize_adaptive(image, text, target_height, variation=1.0, opacity_target=.82, strength=.65, *,
                       x_height=None, measurement=None, calibration=None):
    if not .5<=opacity_target<=.95 or not 0<=strength<=1:
        raise ValueError('Opacity target must be .5–.95 and ink strength 0–1')
    measured=measurement or measure(image,text)
    box=measured['box'];crop=white_image(image).crop(box)
    raw_baseline=measured['baseline'];method=measured['method']
    observed_body=measured['x_height']
    if not measured['confidence'] and calibration:
        observed_body=crop.height*calibration['fractions'][measured['category']]
        method+=';page-calibrated-fallback'
    target_body=float(x_height if x_height is not None else target_height*.73)
    if not np.isfinite(target_body) or target_body<3:raise ValueError('Target x-height must be at least 3 pixels')
    variation=float(np.clip(variation,.95,1.05))
    scale=target_body*variation/observed_body
    height=max(3,round(crop.height*scale))
    width = max(1, round(crop.width * height / crop.height))
    a = np.asarray(crop, dtype=float)
    background = max(235., float(np.quantile(a, .98)))
    opacity = np.clip((background - a) / background, 0, 1)
    ink = opacity[opacity > .25]
    current = float(np.quantile(ink, .70)) if len(ink) else .82
    requested = opacity_target / max(current, .1)
    gain = 1 + float(np.clip(strength, 0, 1)) * (float(np.clip(requested, .82, 1.18)) - 1)
    opacity *= gain
    # Only pale background is removed. No dilation/erosion/thinning/sharpening.
    fade = np.clip((opacity - .04) / .08, 0, 1)
    opacity *= fade
    pixels = np.round(255 * (1 - np.clip(opacity, 0, 1))).astype('uint8')
    crop = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    baseline = max(0, min(height - 1, round(raw_baseline * height / (box[3] - box[1]))))
    return crop, baseline, {'raw_ink_box': list(box), 'normalized_height': height, 'normalized_width': width,
        'baseline_method': method, 'observed_body_fraction': observed_body/(box[3]-box[1]), 'estimated_baseline_raw': raw_baseline,
        'estimated_x_height_raw':observed_body,'target_x_height':target_body,'estimated_x_height_rendered':observed_body*height/(box[3]-box[1]),
        'x_height_reliable':measured['confidence'],'uniform_scale':height/(box[3]-box[1]),'size_variation_factor':variation,
        'ink_gain': gain, 'opacity_target': opacity_target, 'processing': 'adaptive-xheight-v2',
        'note': 'Luminance/scale harmonization, not spelling repair or physical stroke-thickness equalization.'}
