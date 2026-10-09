"""Finalize human-reviewed findings and measured lexical OCR metrics."""
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/evaluation'
report=json.loads((OUT/'evaluation_report.json').read_text())
assert report['completed'] and len(report['comparisons'])==3

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

normal=lambda s:re.sub('[^a-z0-9]','',s.lower())
summary=[]
for i,entry in enumerate(report['comparisons'],1):
    assert entry['sentence'] in [
        'This is my first generated handwriting test.',
        'Artificial intelligence is changing the world.',
        'The velocity and acceleration change over time.']
    for label,data in entry['variants'].items():
        image=ROOT/data['output']
        assert hashlib.sha256(image.read_bytes()).hexdigest()==data['sha256']
        with Image.open(image) as im:im.verify()
        errors=0;characters=0;correct=0
        for row in data['ocr_words']:
            target,recognized=normal(row['target']),normal(row['recognized'])
            row['normalized_target']=target;row['normalized_recognized']=recognized
            errors+=edit_distance(target,recognized);characters+=len(target);correct+=target==recognized
        data['lexical_character_error_rate']=errors/characters
        data['lexical_word_accuracy']=correct/len(data['ocr_words'])
        data['lexical_character_errors']=errors;data['lexical_target_characters']=characters
        data['lexical_correct_words']=correct
        if 'word_positions' in data:
            lines={}
            for p in data['word_positions']:lines.setdefault(p['line'],[]).append(p)
            gaps=[]
            for line in lines.values():
                line.sort(key=lambda p:p['x'])
                gaps.extend(b['x']-(a['x']+a['width']) for a,b in zip(line,line[1:]))
            data['compositor_word_gaps_pixels_before_2x_upscale']=gaps
            data['line_count']=len(lines)
        (OUT/f'{label}_{i}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    with Image.open(OUT/f'comparison_{i}.png') as im:im.verify()
    summary.append({'sentence':i,**{label:{'word_accuracy':data['lexical_word_accuracy'],
        'character_error_rate':data['lexical_character_error_rate']} for label,data in entry['variants'].items()},
        'pixel_difference':entry['paired_pixel_difference']})

report['ocr_normalization']='Within each known word region, lowercase and retain only ASCII letters/digits. Ignore OCR-added whitespace and punctuation. These are lexical word/CER measures, not exact-sentence punctuation or case accuracy.'
report['limitations'].extend([
    'No genuine same-text handwriting ground truth exists for these sentences; writer similarity is qualitative only.',
    'First pair metadata was recovered from hand-inspected final-PNG word regions after serialization failed; pairs 2 and 3 use compositor input word images. Both adapters use the same OCR method within each pair. Avoid treating across-sentence differences as a standardized benchmark.',
    'CV layout scores are uncalibrated repository heuristics, not writer-style measures; reported compositor gaps are in pre-upscale pixels.',
    'Changing was read as charging, This as Thats, and time as firmee by TrOCR. These are OCR observations, not definitive assertions about the intended letter sequence.',
    'Generation time varied widely; runtime is not evidence of model quality.'
])
report['settings'].update({'deterministic_algorithms':True,'cuda_tf32':False,'cudnn_benchmark':False,
                           'cublas_workspace_config':':4096:8'})
report['environment_versions']={name:version(name) for name in ['torch','torchvision','transformers','diffusers','huggingface-hub','Pillow']}
report['visual_findings']={
    'handwriting_similarity':'No clear improvement after 30 steps. Both outputs share broadly the same rounded and sometimes connected forms, but differ from the more regular genuine reference in size, slant and connections. Minor stroke changes do not establish better writer fidelity.',
    'readability_and_correctness':'Most words are visually recognizable. The damaged This in pair 1, stitched intelligence in pair 2 and rough time in pair 3 persist. Normalized lexical OCR results are identical within every pair; no observed text-quality reduction or improvement.',
    'letter_shapes_and_connections':'The fine-tuned adapter slightly changes individual strokes and thickness. Major shapes and the chunk-joining defects remain similar. No systematic improvement in connections is visible.',
    'word_spacing':'Paired outputs retain nearly the same horizontal spacing and line wrapping. Handwriting/test in pair 1 are closely spaced; pair 3 has large size differences between change and over. There is no consistent spacing improvement.',
    'baseline_consistency':'Pair 2 retains the uneven size and vertical alignment of stitched intelligence. Pair 3 remains five lines with variable word heights. The fine-tuned version does not visibly improve baseline regularity.',
    'overfitting_or_memorization':'No obvious copied multiword training sequence is visible. Most requested words were absent from the 18-word pilot, and the outputs resemble their pretrained counterparts. Three seeded examples with training references and overlapping source pages cannot establish absence of overfitting.',
    'conclusion':'This small style-projection pilot did not demonstrate better handwriting similarity and did not show a lexical readability regression on the three comparisons.'}
report['improvement_supported']=False
report['text_quality_reduction_observed']=False
report['metric_summary']=summary
for label,relative in report['checkpoint_paths'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==report['checkpoint_sha256'][label]
assert hashlib.sha256((ROOT/'outputs/erdem_test.png').read_bytes()).hexdigest()=='39fdb9b7c25435f9da31868de95869deca00392eed044a298792c9cf3db0a012'
(OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('Three panels, six generated PNGs, checkpoint hashes and Step 5 baseline verified.')
