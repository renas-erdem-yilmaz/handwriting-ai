"""Real candidate experiment with paired processing comparison and preservation hashes."""
import time
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from .common import ROOT,read_json,write_json,sha
from .layout import tokens
from .quality import prepare,render_session

OUT=ROOT/'outputs/quality_tests'

def panels(paths,labels,target):
    if target.exists():raise FileExistsError(target)
    images=[Image.open(ROOT/path).convert('RGB') for path in paths]
    w=max(im.width for im in images);h=max(im.height for im in images)
    ref=Image.open(ROOT/'data/crops/page_001_region_011_crop_0011.png').convert('RGB')
    ref.thumbnail((w*len(images)-40,100))
    canvas=Image.new('RGB',(w*len(images),h+210),'white');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    d.text((20,12),'GENUINE scanned handwriting reference (not generated)',font=font,fill='black');canvas.paste(ref,(20,42))
    for n,(im,label) in enumerate(zip(images,labels)):
        d.text((n*w+20,156),label,font=font,fill='black');canvas.paste(im,(n*w,210))
    canvas.save(target)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected_file=OUT/'protected_before.json'
    if protected_file.exists():protected=read_json(protected_file)
    else:
        paths=[p for p in (ROOT/'data').rglob('*') if p.is_file()]
        paths += [p for directory in ['outputs/layout_tests','outputs/evaluation','outputs/finetune/erdem_pilot'] for p in (ROOT/directory).rglob('*') if p.is_file()]
        paths += [ROOT/'outputs/erdem_test.png']
        protected={str(p.relative_to(ROOT)):sha(p) for p in paths};write_json(protected_file,protected)
    # Use known existing seeds for candidate 1 where settings/model match exactly.
    doc=read_json(ROOT/'page_engine/examples/quality.json')
    choices=defaultdict(list)
    for name in ['newton','physics','notes']:
        for word in read_json(ROOT/f'outputs/layout_tests/{name}.json')['word_records']:
            choices[word['text']].append(word['seed'])
    used=defaultdict(int);seeds={};idx=0
    for block in doc['blocks']:
        if block['kind']=='math':continue
        for source,text in tokens(block['text']):
            if source!='word':continue
            if choices[text]:
                seeds[str(idx)]=choices[text][used[text]%len(choices[text])];used[text]+=1
            idx+=1
    doc['word_seeds']=seeds
    document_path=OUT/'document.json'
    if document_path.exists():assert read_json(document_path)==doc
    else:write_json(document_path,doc)
    start=time.monotonic()
    session,store=prepare('outputs/quality_tests/document.json','outputs/quality_tests/review_session.json','page_engine/styles/erdem.json')
    # Both comparisons use exactly the same first candidate for every occurrence.
    # Manual decisions can be incorporated later with a separate reviewed prefix.
    before=render_session(session,store,'outputs/quality_tests/before',legacy=True)
    after=render_session(session,store,'outputs/quality_tests/after_processing')
    assert [w['cache_id'] for w in before['word_records']]==[w['cache_id'] for w in after['word_records']]
    panels([before['pages'][0],after['pages'][0]],['BEFORE: Step 9 processing','AFTER: adaptive ink / body height / baseline'],OUT/'comparison_processing.png')
    for name,expected in protected.items():assert sha(ROOT/name)==expected,name
    for image in OUT.glob('*.png'):
        with Image.open(image) as im:im.verify()
    import torch
    report={'completed':False,'stage':'Generated; visual review and interface tests pending',
            'model':session['model'],'style_config':session['style_config'],'sampling':session['sampling'],
            'candidate_count_per_word':3,'word_occurrences':len(session['words']),
            'candidate_bundles':len(set(w['bundle_id'] for w in session['words'])),
            'before':before['pages'],'after_processing':after['pages'],
            'comparison':'outputs/quality_tests/comparison_processing.png',
            'same_raw_words_verified':True,'same_math_layout_text_and_style':True,
            'comparison_note':'Same candidate images, text, style refs, DDIM settings and seed pool. Only normalization and bounded layout variation differ. No OCR score or writer-similarity score is invented.',
            'new_candidate_inference':store.generated,'cache_hits':store.hits,'seconds':time.monotonic()-start,
            'gpu':torch.cuda.get_device_name() if store.pipeline.ready and store.pipeline.device=='cuda' else None,
            'peak_allocated_gpu_gib':torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0,
            'protected_files_unchanged':len(protected),'training_performed':False,'mandatory_ocr':False}
    write_json(OUT/'quality_report.json',report)
    print('Paired real previews saved; all protected hashes unchanged.',flush=True)

if __name__=='__main__':main()
