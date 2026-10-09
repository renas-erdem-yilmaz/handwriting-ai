"""Audit the original CSVs and build only explicitly reviewed word crops."""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/finetune_words'
REPORTS = ROOT / 'training/reports'
MULTILINE_OBSERVATIONS = {
    'page_001_region_002_crop_0002.png': 'Main line plus part of following line visible at bottom.',
    'page_001_region_003_crop_0003.png': 'Main line plus ink from previous line visible at top.',
    'page_001_region_006_crop_0006.png': 'Two text lines and an arrow.',
    'page_001_region_018_crop_0018.png': 'Equation, arrow and multiple text lines.',
    'page_002_region_002_crop_0042.png': 'Main line plus previous line visible at top.'
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    REPORTS.mkdir(parents=True, exist_ok=True)
    vocab = set(json.loads((ROOT / 'diffusionpen-training/letter2index.json').read_text()))
    reviewed = json.loads((ROOT / 'training/reviewed_words.json').read_text())['sources']
    lookup = {(s['split'], s['filename']): s for s in reviewed}
    originals, rows, manifest = {}, [], []
    all_samples = {}
    for split in ('train', 'val', 'test'):
        csv_path = ROOT / 'data/dataset' / split / 'labels.csv'
        originals[str(csv_path.relative_to(ROOT))] = sha(csv_path)
        samples = list(csv.DictReader(csv_path.open(encoding='utf-8-sig', newline='')))
        all_samples[split] = samples
        for sample in samples:
            text = sample['text']
            path = csv_path.parent / 'images' / sample['filename']
            originals[str(path.relative_to(ROOT))] = sha(path)
            with Image.open(path) as im:
                w, h = im.size
            unsupported = sorted(set(text) - vocab)
            # All line images remain held for manual word-to-text alignment.
            status = 'reviewed_subset' if (split, sample['filename']) in lookup else 'pending_word_review'
            rows.append({**sample, 'split': split, 'width': w, 'height': h,
                         'label_word_count': len(text.split()), 'label_has_newline': '\n' in text,
                         'visual_multiline_check': MULTILINE_OBSERVATIONS.get(sample['filename'],
                             'Reviewed single-word boxes only' if (split,sample['filename']) in lookup else 'Pending visual line/word review'),
                         'unsupported': [{'char': c, 'codepoint': f'U+{ord(c):04X}', 'name': unicodedata.name(c, 'UNKNOWN')} for c in unsupported],
                         'math_suspected': sample['type'] in ('mixed', 'equation') or bool(re.search(r'[=|^αβθφϕ→⇒]', text)),
                         'fraction_or_division_notation': '/' in text,
                         'spatial_math_requires_review': sample['type'] in ('mixed','equation') or '/' in text,
                         'status': status})
        split_out = OUT / split
        (split_out / 'images').mkdir(parents=True, exist_ok=True)
        split_words = []
        for source in [s for s in reviewed if s['split'] == split]:
            parent = next(s for s in samples if s['filename'] == source['filename'])
            if set(parent['text']) - vocab:
                raise ValueError(f'Reviewed source has unsupported characters: {parent}')
            image = Image.open(csv_path.parent / 'images' / source['filename']).convert('L')
            for i, (text, box) in enumerate(source['words']):
                assert text in parent['text'].split(), (text, parent['text'])
                assert 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
                arr = np.array(image.crop(box))
                # Preserve faint ink in page 2; use a conservative background
                # cutoff rather than erasing gray handwritten strokes.
                arr[arr > 205] = 255
                ink = np.argwhere(arr < 205)
                assert len(ink) > 10, (source['filename'], text)
                y0, x0 = ink.min(0); y1, x1 = ink.max(0) + 1
                word = Image.fromarray(arr[y0:y1, x0:x1])
                # Preserve aspect ratio; center-pad to official 64x256 canvas.
                scale = min(60 / word.height, 248 / word.width)
                word = word.resize((max(1, round(word.width * scale)), max(1, round(word.height * scale))), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (256, 64), 'white')
                canvas.paste(word, ((256-word.width)//2, (64-word.height)//2))
                name = f"{Path(source['filename']).stem}_word_{i:02d}.png"
                canvas.save(split_out / 'images' / name)
                record = {'filename': name, 'text': text, 'type': 'word', 'page': parent['page'], 'writer': 'erdem',
                          'parent_filename': source['filename'], 'box': box, 'split': split, 'reviewed': True}
                split_words.append(record); manifest.append(record)
        fields = ['filename', 'text', 'type', 'page', 'writer', 'parent_filename']
        with (split_out / 'labels.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fields); writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in split_words)
        sheet = Image.new('RGB', (3*276, ((len(split_words)+2)//3)*100), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(split_words):
            x, y = (i%3)*276, (i//3)*100
            sheet.paste(Image.open(split_out/'images'/r['filename']), (x, y+22))
            draw.text((x+4,y+3), f"{i}: {r['text']}", fill='black')
        sheet.save(OUT / f'{split}_review.png')
    ids = [set(r['filename'] for r in all_samples[s]) for s in ('train','val','test')]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]), 'Original split leakage'
    page_sets = {s: sorted(set(r['page'] for r in rs)) for s,rs in all_samples.items()}
    summary = {'original_split_counts': {s: len(rs) for s,rs in all_samples.items()},
               'reviewed_word_counts': dict(Counter(r['split'] for r in manifest)),
               'unsupported_sample_counts': dict(Counter(r['split'] for r in rows if r['unsupported'])),
               'multiword_sample_count': sum(r['label_word_count']>1 for r in rows),
               'pages_by_split': page_sets,
               'reviewed_pages_by_split': {s:sorted(set(r['page'] for r in manifest if r['split']==s)) for s in ('train','val','test')},
               'visually_identified_line_contamination': MULTILINE_OBSERVATIONS,
               'limitations': ['Only manually reviewed word boxes are used. Other samples explicitly remain pending; no inferred alignment.',
                              'All splits share source pages: region-level holdout, not independent page-level evaluation.',
                              'Only one writer and a tiny reviewed pilot: validation cannot establish broad generalization.',
                              'Step 5 style reference came from test data; do not use it for training or unbiased test claims.',
                              'Long words, equations and multiline crops require further review before expansion.']}
    for name, value in [('label_audit.json', rows), ('unsupported_samples.json',[r for r in rows if r['unsupported']]),
                        ('word_manifest.json',manifest), ('data_summary.json',summary), ('original_sha256.json',originals)]:
        (REPORTS/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    with (REPORTS/'unsupported_samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['split','filename','text','unsupported_codepoints','status']
        writer=csv.DictWriter(f,fields); writer.writeheader()
        for r in rows:
            if r['unsupported']:
                writer.writerow({**{k:r[k] for k in ('split','filename','text','status')},
                                 'unsupported_codepoints':'; '.join(f"{c['char']} {c['codepoint']} {c['name']}" for c in r['unsupported'])})
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    prepare()
