"""Record the observed experiment, provisional selections and final integrity checks."""
from PIL import Image
from .common import ROOT,read_json,write_json,sha
from .quality import open_store,render_session
from .quality_experiment import panels

OUT=ROOT/'outputs/quality_tests'

def main():
    session=read_json(OUT/'review_session.json');store=open_store(session)
    before=read_json(OUT/'before.json');after=read_json(OUT/'after_final.json')
    assert [w['cache_id'] for w in before['word_records']]==[w['cache_id'] for w in after['word_records']]
    assert before['generation']==after['generation'] and before['model']==after['model']
    panels([before['pages'][0],after['pages'][0]],['BEFORE: Step 9 processing','AFTER: same raw words, adaptive processing'],OUT/'comparison_final.png')
    bundle_ids={w['bundle_id'] for w in session['words']}
    counts={key:len(store.load(key)['candidates']) for key in bundle_ids}
    target=session['words'][16]['bundle_id']
    assert counts[target]==4 and all(n==3 for key,n in counts.items() if key!=target)
    # Writer approval is not simulated. These visual choices remain pending.
    choices={0:2,1:2,5:1,28:2,34:2,39:2}
    for index,chosen in choices.items():
        item=session['words'][index]
        store.decide(item['bundle_id'],chosen,'pending','Assistant provisional visual choice: cleaner letter joins/less heavy strokes. Writer approval pending.')
    for index in [16,26,37]:
        item=session['words'][index]
        store.decide(item['bundle_id'],status='problematic',note='Acceleration candidates still have ambiguous, duplicated or merged letters. Structural flags are insufficient; human review/regeneration needed.')
    reviewed=render_session(session,store,'outputs/quality_tests/after_review')
    panels([before['pages'][0],reviewed['pages'][0]],['BEFORE: first candidates, legacy processing','AFTER: provisional choices; long words still problematic'],OUT/'comparison_review.png')
    assert not store.pipeline.ready and store.generated==0
    try:render_session(session,store,'outputs/quality_tests/approval_gate_test',require_approved=True)
    except ValueError as exc:assert 'needs approval' in str(exc)
    else:raise AssertionError('Approval gate failed')
    for name,expected in read_json(OUT/'protected_before.json').items():assert sha(ROOT/name)==expected,name
    assert sha(store.pipeline.base_path)==session['model']['base_sha256']
    assert sha(store.pipeline.style_path)==session['model']['style_encoder_sha256']
    for key in bundle_ids:
        for candidate in store.load(key)['candidates']:assert sha(ROOT/candidate['image'])==candidate['sha256']
    for path in OUT.glob('*.png'):
        with Image.open(path) as image:image.verify()
    report=read_json(OUT/'quality_report.json')
    report.update(completed=True,stage='Generated, visually inspected, interface and integrity checks passed',
        after_final=after['pages'],after_provisional_review=reviewed['pages'],
        comparison='outputs/quality_tests/comparison_final.png',review_comparison='outputs/quality_tests/comparison_review.png',
        human_approved_word_count=0,provisional_choice_occurrences=choices,
        candidate_counts_by_bundle=counts,total_distinct_candidates=sum(counts.values()),
        manual_regeneration_test={'word_index':16,'text':'acceleration.','seed':555555,'new_candidates':1,
                                  'other_bundles_still_three_candidates':True,'previous_candidates_preserved':True,
                                  'warning':'The extra candidate remains distorted; regeneration is not a spelling guarantee.'},
        tests={'step9_regression_tests':4,'quality_tests':3,'passed':True,
               'tkinter_fixture_buttons_regeneration_and_resume':True,'real_session_tkinter_startup_navigation_and_fourth_candidate':True,
               'viewing_and_rendering_without_gpu_model_load':True,'require_approved_gate':True,
               'raw_candidate_sha256_verification':True,'protected_originals_and_pretrained_checkpoints':True},
        findings={
            'readability':'Processing alone preserves the same malformed letters. Provisional changes/rest/title candidates have cleaner joins, but acceleration remains problematic in all sampled variants. No OCR or spelling certification is claimed.',
            'ink_consistency':'Pale background texture and excessive dark/light opacity differences are reduced modestly. Thick force/mass strokes still differ substantially from thin a/of/find strokes; geometry is not equalized.',
            'letter_height':'Body-aware scaling gives descender/ascender words more comparable body size than fixed whole-box height. Short words and unusual flourishes can still confuse the estimate; aspect ratios are preserved.',
            'baseline_alignment':'Ink-column body estimates place descenders below a common line baseline, with bounded vertical jitter. This is more systematic than per-word whole-box alignment but not perfect for malformed glyphs.',
            'word_spacing':'Configured gaps and actual-width wrapping remain orderly. Changing size or candidate widths can move line breaks; paragraph separation is preserved.',
            'writer_resemblance':'No established improvement in personal writer identity. Several generated words remain heavier and more printed than the genuine scan; preprocessing and candidate selection do not replace writer conditioning.'},
        limitations=['No retraining; pretrained default and optional adapter unchanged.',
                     'Only five training-only style crops condition a stochastic model; style drift remains.',
                     'Long-word chunk stitching can introduce extra/missing/merged letters.',
                     'OCR is optional and was not used in this experiment.',
                     'All real candidates remain pending or problematic until the writer approves them.',
                     'Review one session with a single writing process at a time.',
                     'Mathematics and synthesized punctuation are not genuine handwritten writer glyphs.'],
        recommended_next_steps=['Review the supplied session, choosing candidates for readability and a common pen appearance.',
                                'Regenerate or mark problematic when all alternatives are malformed; do not approve on appearance alone.',
                                'Try a different coherent set of five clean word references through a copied style config; compare before making style claims.',
                                'Use optional OCR as a secondary lexical check, never as proof of writer resemblance.'])
    report['final_images_sha256']={str(path.relative_to(ROOT)):sha(path) for path in OUT.glob('*.png')}
    write_json(OUT/'quality_report.json',report)
    print('Completed quality report; no model inference during final composition. Actual writer approvals: 0.')

if __name__=='__main__':main()
