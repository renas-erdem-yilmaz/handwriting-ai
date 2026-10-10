import numpy as np
from PIL import Image

def ink_box(image, threshold=190):
    a = np.asarray(image.convert('L'))
    ys, xs = np.where(a < threshold)
    if not len(xs):
        raise ValueError('No visible ink in word')
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def normalize(image, text, target_height, variation=1.0):
    """Legacy whole-ink-height scaling; default layout uses adaptive x-height."""
    box = ink_box(image)
    crop = image.convert('L').crop(box)
    # Suppress pale diffusion background texture without altering dark ink.
    # Raw cached candidates remain intact for review and future normalization.
    pixels = np.asarray(crop).copy()
    pixels[pixels > 210] = 255
    crop = Image.fromarray(pixels)
    # Short lowercase words have no ascenders; do not inflate them to capital height.
    small = text.strip('.,;:!?').islower() and not any(c in text for c in 'bdfghjklpqty')
    height = max(8, round(target_height * (0.73 if small else 1.0) * np.clip(variation, .94, 1.06)))
    width = max(1, round(crop.width * height / crop.height))
    crop = crop.resize((width, height), Image.Resampling.LANCZOS)
    # Reuse the ink-profile baseline detector, with lexical descender information.
    from reforge.compose.layout import detect_baseline
    baseline = int(detect_baseline(np.asarray(crop), text))
    return crop, max(0, min(height - 1, baseline)), {'raw_ink_box': list(box), 'normalized_height': height, 'normalized_width': width}

def assess(image, text):
    """Structural screening only: this does NOT verify spelling."""
    try:
        box = ink_box(image)
    except ValueError:
        return ['blank'], 100.0
    a = np.asarray(image.convert('L').crop(box))
    ratio = a.shape[1] / a.shape[0]
    density = float((a < 190).mean())
    flags = []
    if a.shape[0] < 6: flags.append('very_small_ink')
    if density < .025 or density > .65: flags.append('unusual_ink_density')
    n = len(text.strip('.,;:!?'))
    if ratio > max(3, n * 1.5) or (n > 4 and ratio < .7): flags.append('unusual_aspect_ratio')
    if box[0] == 0 or box[2] == image.width: flags.append('ink_touches_horizontal_edge')
    return flags, float(len(flags))
