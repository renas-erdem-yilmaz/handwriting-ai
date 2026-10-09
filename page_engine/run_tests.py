"""Run the remaining real previews, cache/regeneration checks and pagination demo."""
import time
import sys
from collections import defaultdict
from PIL import Image
from .common import ROOT, read_json, write_json, sha
from .model import Pipeline
from .words import WordStore, review
from .layout import build, tokens

def main():
    start = time.monotonic()
    if (ROOT / 'outputs/layout_tests/test_report.json').exists():
        raise FileExistsError('Completed test report exists; preserve it')
    protected_paths = [p for p in (ROOT / 'data').rglob('*') if p.is_file()]
    protected_paths += [p for p in (ROOT / 'outputs/evaluation').glob('*') if p.is_file()]
    protected_paths += [ROOT / 'outputs/erdem_test.png', *list((ROOT / 'outputs/finetune/erdem_pilot').glob('*.pt'))]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    pipeline = Pipeline()
    protected[str(pipeline.base_path.relative_to(ROOT))] = sha(pipeline.base_path)
    protected[str(pipeline.style_path.relative_to(ROOT))] = sha(pipeline.style_path)
    store = WordStore(pipeline, {'steps': 20, 'guidance': 3.0, 'retries': 1, 'ocr': False})
    profile = read_json(ROOT / 'page_engine/profiles/scanned.json')
    results = {}
    for example in ['physics', 'notes']:
        existing = ROOT / f'outputs/layout_tests/{example}.json'
        if existing.exists():
            result = read_json(existing)
            assert result['model'] == pipeline.identity and result['generation'] == store.settings
            assert result['document'] == read_json(ROOT / f'page_engine/examples/{example}.json')
        else:
            result = build(read_json(ROOT / f'page_engine/examples/{example}.json'), store, f'outputs/layout_tests/{example}', profile)
        results[example] = {'pages': result['pages'], 'word_count': len(result['word_records']), 'math_runs': sum(p['rendering']['source'] == 'typeset_math_not_handwriting' for p in result['positions'])}
    newton = read_json(ROOT / 'outputs/layout_tests/newton.json')
    repeated = [r for r in newton['word_records'] if r['text'] == 'force']
    repeated_hashes = [read_json(ROOT / '.reforge-cache/words' / (r['cache_id'] + '.json'))['image_sha256'] for r in repeated]
    results['natural_variation'] = {'word': 'force', 'occurrences': len(repeated), 'distinct_images': len(set(repeated_hashes))}
    assert len(set(repeated_hashes)) > 1
    # Cache-only repeated-body stress test: replay distinct real generated candidates.
    pool = defaultdict(list)
    for record in newton['word_records']: pool[record['text']].append(record)
    body = next(b for b in newton['document']['blocks'] if b['kind'] == 'paragraph')
    long_doc = {'seed': 9100, 'layout': {'page': 'notebook'}, 'blocks': [dict(body) for _ in range(18)]}
    occurrence = defaultdict(int); overrides = {}; idx = 0
    for block in long_doc['blocks']:
        for source, text in tokens(block['text']):
            if source == 'word':
                candidates = pool[text]; selected = candidates[occurrence[text] % len(candidates)]
                overrides[str(idx)] = selected['seed']; occurrence[text] += 1; idx += 1
    before = store.generated
    multi = build(long_doc, store, 'outputs/layout_tests/multipage', profile, overrides)
    assert len(multi['pages']) >= 2 and before == store.generated
    results['multipage'] = {'pages': multi['pages'], 'word_count': len(multi['word_records']), 'no_new_inference': True,
                            'note': 'Layout stress test repeats genuine generated words from Newton preview; it is not new-model inference.'}
    # Actual regeneration creates one new cached candidate, leaving original intact.
    old = newton['word_records'][0]
    image, changed = store.obtain(old['text'], 123456)
    assert changed['id'] != old['cache_id']
    original = read_json(ROOT / '.reforge-cache/words' / (old['cache_id'] + '.json'))
    results['regeneration'] = {'word': old['text'], 'old_id': old['cache_id'], 'new_id': changed['id'],
                               'different_pixels': original['image_sha256'] != changed['image_sha256']}
    old_status = changed['status']
    assert review(changed['id'], 'approved')['status'] == 'approved'
    review(changed['id'], old_status)  # API persistence test only, not a human approval.
    revised = build(newton['document'], store, 'outputs/layout_tests/newton_regenerated', profile, {'0':123456})
    results['regeneration']['revised_pages'] = revised['pages']
    results['regeneration']['approval_persistence_test'] = True
    # Exactly matched request must be a cache hit and must not call the generator.
    old_word_method = pipeline.word
    pipeline.word = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Cache should avoid inference'))
    cached, record = store.obtain(old['text'], old['seed'])
    pipeline.word = old_word_method
    assert record['id'] == old['cache_id']
    for name, expected in protected.items(): assert sha(ROOT / name) == expected, name
    for manifest_name in ['newton', 'physics', 'notes', 'multipage', 'newton_regenerated']:
        manifest = read_json(ROOT / f'outputs/layout_tests/{manifest_name}.json')
        bounds = manifest['geometry']['usable_box']
        for pos in manifest['positions']:
            x,y,r,b = pos['box']; left,top,right,bottom = bounds
            assert left <= x < r <= right and top <= y < b <= bottom
        for path in manifest['pages']:
            with Image.open(ROOT / path) as image: image.verify()
    import torch
    report = {'completed': True, 'pretrained_default': True, 'training_performed': False, 'device': pipeline.device,
              'gpu': torch.cuda.get_device_name() if pipeline.device == 'cuda' else None,
              'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3 if pipeline.device == 'cuda' else 0,
              'seconds': time.monotonic() - start, 'model': pipeline.identity, 'checkpoint': str(pipeline.base_path.relative_to(ROOT)),
              'style_encoder': str(pipeline.style_path.relative_to(ROOT)), 'sampling': store.settings,
              'profile': 'page_engine/profiles/scanned.json', 'tests': results, 'cache_hits': store.hits,
              'new_word_candidates': store.generated, 'cache_hit_without_inference': True,
              'optional_ocr_module_imported': 'reforge.evaluate.ocr' in sys.modules,
              'protected_files_verified': len(protected), 'protected_originals_unchanged': True,
              'limits': ['No guarantee of correct spelling; all word candidates remain pending human review.',
                         'Optional OCR was not enabled for these previews.', 'No writer-similarity improvement is claimed.',
                         '18 training words and shared source pages limit adapter interpretation; pretrained remains default.',
                         'Reforge long-word stitching and contraction artifacts can persist after normalization.',
                         'Mathematics is STIX font rendering, not handwriting; parser syntax is a subset of TeX.']}
    write_json(ROOT / 'outputs/layout_tests/test_report.json', report)
    print('Real previews, regeneration, cache reuse and multi-page checks passed.', flush=True)

if __name__ == '__main__': main()
