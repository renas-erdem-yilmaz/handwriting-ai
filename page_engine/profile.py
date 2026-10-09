"""Conservative scan measurements; never writes to data/."""
import numpy as np
import cv2
from PIL import Image
from .common import ROOT, sha, write_json

def analyze(destination):
    pages = []
    for path in sorted((ROOT / 'data/pages').glob('*.png')):
        image = Image.open(path).convert('L')
        image.thumbnail((1000, 1600))
        a = np.asarray(image)
        mask = (a < 125).astype('uint8')
        # Ignore punched left edge, page boundary, and printing at the bottom.
        mask[:, :round(a.shape[1] * .035)] = 0
        mask[:round(a.shape[0] * .045)] = 0
        mask[round(a.shape[0] * .96):] = 0
        # Remove long printed grid lines while retaining dark word strokes.
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 120), 'uint8'))
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((120, 1), 'uint8'))
        mask = mask & (1 - horizontal) & (1 - vertical)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        for n in range(1, count):
            x, y, w, h, area = stats[n]
            if 5 <= area <= 5000 and 3 <= h <= 90 and w < 200:
                clean[labels == n] = 1
        ys, xs = np.where(clean)
        if not len(xs): continue
        rows = np.convolve(clean.sum(axis=1), np.ones(5) / 5, mode='same')
        active = np.flatnonzero(rows > max(3, np.percentile(rows[rows > 0], 45)))
        bands = np.split(active, np.where(np.diff(active) > 6)[0] + 1)
        centers = [float(np.mean(b)) for b in bands if len(b) >= 4]
        distances = np.diff(centers)
        usual = distances[(distances > .009 * a.shape[0]) & (distances < .045 * a.shape[0])]
        pitch = float(np.median(usual) / a.shape[0]) if len(usual) else .024
        gaps = distances[(distances >= .045 * a.shape[0]) & (distances < .1 * a.shape[0])]
        first = clean[:round(a.shape[0] * .14)]
        hy, hx = np.where(first)
        pages.append({'page': str(path.relative_to(ROOT)), 'sha256': sha(path), 'original_size': list(Image.open(path).size),
                      'margins_fraction': [float(np.percentile(xs, 1) / a.shape[1]), float(1 - np.percentile(xs, 99) / a.shape[1]),
                                           float(np.percentile(ys, 1) / a.shape[0]), float(1 - np.percentile(ys, 99) / a.shape[0])],
                      'line_pitch_fraction': pitch, 'paragraph_gap_fraction': float(np.median(gaps) / a.shape[0]) if len(gaps) else 2 * pitch,
                      'heading_center_x_fraction': float(np.median(hx) / a.shape[1]) if len(hx) else .5,
                      'heading_top_fraction': float(hy.min() / a.shape[0]) if len(hy) else .08,
                      'detected_line_bands': len(centers)})
    if not pages: raise ValueError('No usable scanned pages')
    margins = np.median([p['margins_fraction'] for p in pages], axis=0)
    profile = {'schema': 1, 'source': 'Heuristic dark-ink analysis of scanned pages, excluding faint grid and page edges',
               'page_count': len(pages), 'pages': pages,
               'margins_fraction': np.clip(margins, [.055, .035, .07, .04], [.15, .15, .2, .15]).tolist(),
               'line_pitch_fraction': float(np.clip(np.median([p['line_pitch_fraction'] for p in pages]), .024, .04)),
               'paragraph_extra_lines': 1.0, 'heading_align': 'center',
               'measured_heading_center_x_fraction': float(np.median([p['heading_center_x_fraction'] for p in pages])),
               'measured_heading_top_fraction': float(np.median([p['heading_top_fraction'] for p in pages])),
               'measured_paragraph_gap_fraction': float(np.median([p['paragraph_gap_fraction'] for p in pages])),
               'limitations': ['Diagrams, boxed equations and question numbers affect these estimates.',
                              'Heading classification is heuristic, not semantic recognition; center alignment is a configurable template choice.',
                              'Measured irregular paragraph gaps are retained; rendering uses a conservative one-line extra gap.',
                              'Margins and line pitch are clamped to usable page ranges; per-page raw measurements are retained.']}
    write_json(destination, profile)
    return profile
