"""Width-aware pagination with explicit baselines and provenance."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import ROOT, read_json, write_json
from .normalize import normalize
from .math_symbols.renderer import render as render_math
from .consistency import normalize_adaptive
from .writing_scale import measure,calibrate

def tokens(text):
    """$...$ is an indivisible inline equation; surrounding prose is word-tokenized."""
    if text.count('$') % 2: raise ValueError('Unmatched math dollar delimiter')
    chunks = text.split('$')
    result = []
    for n, chunk in enumerate(chunks):
        if n % 2:
            if not chunk.strip(): raise ValueError('Empty inline math')
            result.append(('math', chunk))
        else:
            result.extend(('word', word) for word in chunk.split())
    return result

def build(document, store, output_prefix, profile, overrides=None, require_approved=False):
    settings = document.get('layout', {})
    processing = dict(getattr(store, 'processing', {}), **document.get('processing', {}))
    processing.setdefault('mode','adaptive')
    size_variation = float(settings.get('size_variation', .05))
    vertical_px = round(float(settings.get('vertical_variation_mm', .17)) * settings.get('dpi', 150) / 25.4)
    if not 0 <= size_variation <= .05 or not 0 <= vertical_px <= 3:
        raise ValueError('Natural variation must stay within 5% size and 3 pixels vertical')
    dpi = settings.get('dpi', 150)
    kind = settings.get('page', 'a4')
    if kind not in ('a4', 'letter', 'notebook'): raise ValueError('Page must be a4, letter or notebook')
    background=settings.get('background','lined' if kind=='notebook' else 'blank')
    if background not in ('blank','lined'):raise ValueError('Background must be blank or lined')
    lined=background=='lined'
    default_mm = {'a4':[210,297],'letter':[215.9,279.4],'notebook':[176,250]}[kind]
    mm = settings.get('size_mm', default_mm)
    width, height = [round(x * dpi / 25.4) for x in mm]
    fractions = profile['margins_fraction']
    margins = settings.get('margins_mm', [fractions[0] * mm[0], fractions[1] * mm[0], fractions[2] * mm[1], fractions[3] * mm[1]])
    left, right_margin, top, bottom = [round(v * dpi / 25.4) for v in margins]
    right, bottom_y = width - right_margin, height - bottom - 30
    pitch = round(settings.get('line_spacing_mm', profile['line_pitch_fraction'] * mm[1]) * dpi / 25.4)
    target = round(settings.get('word_height_mm', 4.6) * dpi / 25.4)
    x_target=float(settings.get('x_height_mm',settings.get('word_height_mm',4.6)*.73))*dpi/25.4
    heading_scale=float(settings.get('heading_size_scale',1.25))
    if not np.isfinite(x_target) or x_target<3 or not 1<=heading_scale<=2:
        raise ValueError('Invalid writing x-height or heading scale')
    gap = round(settings.get('word_spacing_mm', 2.4) * dpi / 25.4)
    indent = round(settings.get('indent_mm', 5) * dpi / 25.4)
    para_gap = round(settings.get('paragraph_extra_lines', profile['paragraph_extra_lines']) * pitch)
    if gap<0 or indent<0 or para_gap<0 or target<=0:
        raise ValueError('Spacing/indentation cannot be negative; word height must be positive')
    if right - left < target * 3 or bottom_y - top < pitch * 2 or pitch < target + 5:
        raise ValueError('Unusable page geometry or line spacing smaller than word height')
    prefix = ROOT / output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = prefix.with_suffix('.json')
    if manifest_path.exists() or prefix.with_name(prefix.name + '_page_001.png').exists():
        raise FileExistsError('Choose a new output prefix; existing preview is preserved')
    prepared={};scale_profile=None
    if processing.get('mode')=='adaptive':
        cursor=0;body_measurements=[]
        for block in document['blocks']:
            if block['kind']=='math':continue
            for source,text in tokens(block['text']):
                if source!='word':continue
                seed=(overrides or {}).get(str(cursor),document.get('seed',42)+cursor*1009)
                raw,record=store.obtain(text,seed)
                if require_approved and record['status']!='approved':raise ValueError(f'Word needs approval: {record["id"]}')
                geometry=measure(raw,text)
                prepared[cursor]=(raw,record,geometry)
                if block['kind']!='heading':body_measurements.append(geometry)
                cursor+=1
        scale_profile=calibrate(body_measurements)
    rng = np.random.default_rng(document.get('seed', 42))
    pages, positions, word_records = [], [], []
    baseline = top + target
    last_bottom = top - 8
    def new_page():
        page = Image.new('RGB', (width, height), 'white')
        d = ImageDraw.Draw(page)
        if lined:
            for y in range(top + target, bottom_y, pitch): d.line((left, y + 3, right, y + 3), fill=(218, 229, 240), width=1)
            d.line((left - 12, top, left - 12, bottom_y), fill=(235, 202, 202), width=1)
        pages.append(page)
    new_page()
    index = 0
    overrides = overrides or {}
    for block_index, block in enumerate(document['blocks']):
        kind_block = block['kind']
        if kind_block not in ('heading', 'paragraph', 'math'): raise ValueError('Unknown block kind')
        entries = [('math', block['text'])] if kind_block == 'math' else tokens(block['text'])
        atoms = []
        for source, text in entries:
            if source == 'math':
                math_settings = document.get('math', {})
                image, base, meta = render_math(text, dpi, settings.get('math_font_points', 15),
                    library=math_settings.get('library', '.reforge-cache/math_symbols'),
                    mode=math_settings.get('mode', 'auto'),
                    seed=math_settings.get('seed',document.get('seed',42))+block_index*1009+len(atoms),
                    x_height=x_target*float(math_settings.get('size_scale',1)),
                    variation=float(math_settings.get('variation',.025)))
                record = None
            else:
                seed = overrides.get(str(index), document.get('seed', 42) + index * 1009)
                if index in prepared:raw,record,geometry=prepared[index]
                else:raw, record = store.obtain(text, seed)
                if require_approved and record['status'] != 'approved':
                    raise ValueError(f'Word needs approval: {record["id"]}')
                h = target * (heading_scale if kind_block == 'heading' else 1)
                variation = float(rng.uniform(1-size_variation, 1+size_variation))
                if processing.get('mode') == 'adaptive':
                    image, base, meta = normalize_adaptive(raw, text, h, variation,
                        processing.get('opacity_target', .82), processing.get('ink_strength', .65),
                        x_height=x_target*(heading_scale if kind_block=='heading' else 1),measurement=geometry,calibration=scale_profile)
                else:
                    image, base, meta = normalize(raw, text, h, variation)
                meta.update(source='diffusionpen_word', cache_id=record['id'], status=record['status'], flags=record['flags'])
                word_records.append({'index': index, 'text': text, 'seed': seed, 'cache_id': record['id'], 'status': record['status'], 'flags': record['flags'],
                                     **{key:record[key] for key in ('bundle_id','candidate_index') if key in record}})
                index += 1
            usable = right - left - (indent if kind_block == 'paragraph' else 0)
            if image.width > usable:
                if source=='word' and processing.get('mode')=='adaptive':
                    raise ValueError(f'Word too wide at the target writing scale: {text}; enlarge page or reduce x_height_mm. Per-word shrinkage is disabled.')
                # Whole-token reduction is explicit. Never crop or split a mathematical expression.
                ratio = usable / image.width
                if ratio < .55: raise ValueError(f'Token too wide to render legibly: {text}; enlarge page or simplify equation')
                image = image.resize((usable, max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
                base = round(base * ratio); meta['fit_scale'] = ratio
            jitter = int(rng.integers(-vertical_px, vertical_px+1)) if source == 'word' else 0
            atoms.append({'image': image, 'baseline': base, 'text': text, 'meta': meta, 'jitter': jitter})
        # Assemble lines from ACTUAL normalized widths.
        lines, line, used = [], [], (indent if kind_block == 'paragraph' else 0)
        for atom in atoms:
            addition = atom['image'].width + (gap if line else 0)
            if line and used + addition > right - left:
                lines.append(line); line = []; used = 0; addition = atom['image'].width
            line.append(atom); used += addition
        if line: lines.append(line)
        for line_index, line in enumerate(lines):
            ascent = max(a['baseline'] - a['jitter'] for a in line)
            descent = max(a['image'].height - a['baseline'] + a['jitter'] for a in line)
            baseline = max(baseline, top + ascent, last_bottom + 8 + ascent)
            if lined:
                anchor = top + target
                baseline = anchor + max(0, int(np.ceil((baseline - anchor) / pitch))) * pitch
            if baseline + descent > bottom_y:
                new_page(); baseline = top + ascent; last_bottom = top - 8
                if lined:
                    baseline = anchor + max(0, int(np.ceil((baseline - anchor) / pitch))) * pitch
            x = left + (indent if kind_block == 'paragraph' and line_index == 0 else 0)
            heading_align = settings.get('heading_align', profile['heading_align'])
            if kind_block in ('heading', 'math') and heading_align == 'center':
                line_width = sum(a['image'].width for a in line) + gap * (len(line) - 1)
                x = left + ((right - left - line_width) // 2)
            for atom in line:
                image = atom['image']; y = baseline - atom['baseline'] + atom['jitter']
                if not (left <= x and x + image.width <= right and top <= y and y + image.height <= bottom_y):
                    raise AssertionError('Content outside usable page bounds')
                # Multiply luminance through transparent white background to preserve notebook lines.
                region = np.asarray(pages[-1].crop((x, y, x + image.width, y + image.height)), dtype=np.uint16)
                ink = np.asarray(image, dtype=np.uint16)[..., None]
                pages[-1].paste(Image.fromarray((region * ink // 255).astype('uint8')), (x, y))
                positions.append({'page': len(pages), 'block': block_index, 'line': line_index, 'text': atom['text'],
                                  'box': [x, y, x + image.width, y + image.height], 'baseline': baseline,
                                  'rendering': atom['meta']})
                x += image.width + gap
            last_bottom = baseline + descent
            baseline += max(pitch, ascent + descent + 8)
        baseline += para_gap if kind_block == 'paragraph' else round(pitch * .5)
    paths = []
    for n, page in enumerate(pages, 1):
        sources={p['rendering']['source'] for p in positions if p['page']==n and p['rendering']['source']!='diffusionpen_word'}
        if 'typeset_math_not_handwriting' in sources:label='Math: typeset fallback present (NOT handwriting); see manifest.'
        elif 'handwritten_math_symbols' in sources:label='Math: assembled from approved handwritten symbol crops.'
        else:label='Text: DiffusionPen. Preview words need visual review.'
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',max(10,round(dpi*.075)))
        ImageDraw.Draw(page).text((left,height-bottom+3),label,fill=(65,65,65),font=font)
        path = prefix.with_name(prefix.name + f'_page_{n:03d}.png')
        if path.exists(): raise FileExistsError(path)
        page.save(path); paths.append(str(path.relative_to(ROOT)))
    manifest = {'schema': 1, 'document': document, 'model': store.pipeline.identity, 'processing': processing,
                'writing_scale':{'target_x_height_pixels':x_target,'heading_scale':heading_scale,'variation_limit':size_variation,'fallback_calibration':scale_profile},
                'generation': store.settings, 'profile': profile, 'pages': paths, 'word_records': word_records,
                'positions': positions, 'overrides': overrides, 'cache_hits': store.hits, 'new_words': store.generated,
                'geometry': {'size_pixels': [width, height], 'usable_box': [left, top, right, bottom_y], 'line_pitch_pixels': pitch},
                'warnings': ['Preview words require visual review. Structural screening cannot verify spelling.',
                             'Retries are bounded and do not guarantee correct spelling.',
                             'Math provenance and missing approved symbols are recorded per equation. Font fallback is not handwriting.',
                             'Reforge punctuation may use its existing standard-font rendering.']}
    write_json(manifest_path, manifest)
    return manifest
