"""Compose final cleaned previews from cache, then record visual review."""
from PIL import Image
from .common import ROOT, read_json, write_json, sha
from .model import Pipeline
from .words import WordStore
from .layout import build

def main():
    report = read_json(ROOT / 'outputs/layout_tests/test_report.json')
    pipeline = Pipeline()
    store = WordStore(pipeline, report['sampling'])
    profile = read_json(ROOT / 'page_engine/profiles/scanned.json')
    primary = {}
    for example in ['newton', 'physics', 'notes']:
        result = build(read_json(ROOT / f'page_engine/examples/{example}.json'), store,
                       f'outputs/layout_tests/{example}_final', profile)
        assert store.generated == 0, 'Final composition should use cached real words'
        for path in result['pages']:
            with Image.open(ROOT / path) as image: image.verify()
        primary[example] = result['pages']
    report['primary_previews'] = primary
    report['final_composition_cache_hits'] = store.hits
    report['visual_review'] = {
        'newton': 'Title and paragraph wrap into orderly lines; force variants differ. Acceleration has distorted/stitch-like letter shapes. Spelling is not certified.',
        'physics': 'Inline variables and display fractions/superscripts remain readable; equations use distinct standard fonts. Word stroke weight varies; acceleration remains problematic.',
        'notes': 'Requested Greek letters, implication arrow, fractions, superscripts and subscripts render visibly and accurately in STIX. Generated prose is readable but symbols has imperfect letter shapes.',
        'background_cleanup': 'Pale gray diffusion texture removed from composition copies; raw cached candidates preserved.',
        'writer_similarity': 'No claim of improved writer resemblance. These previews use the pretrained base and genuine training-only style references.'}
    report['unit_tests'] = {'count': 4, 'passed': True, 'command': r'.\.reforge\Scripts\python.exe -m page_engine.tests',
                            'coverage': ['bounded retry and cache hit', 'math syntax/Greek/fractions and invalid syntax rejection',
                                         'actual ink normalization', 'width wrapping, pagination, bounds and overwrite refusal']}
    report['dependencies_added'] = ['matplotlib==3.11.2 and its free local dependencies']
    report['optional_ocr_tested'] = False
    report['profile_measured_page_count'] = profile['page_count']
    write_json(ROOT / 'outputs/layout_tests/test_report.json', report)
    print('Final cache-only previews:', primary)

if __name__ == '__main__': main()
