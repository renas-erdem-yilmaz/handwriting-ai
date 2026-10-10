"""Collector, single-equation rendering, and reproducible cached-prose previews."""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .library import Library, DEFAULT_LIBRARY
from .renderer import render
from ..common import ROOT, read_json, write_json, sha

EXAMPLES=['F = ma','a = F/m','v = v_0 + at','x = 1/2 at^2','θ = α + β','v^2 = v_0^2 + 2ax']

def save_equation(expression,output,library,seed=9600):
    path=ROOT/output
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Choose a new output path: '+str(path))
    image,baseline,meta=render(expression,library=library,seed=seed,x_height=19)
    path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
    write_json(path.with_suffix('.json'),dict(meta,baseline=baseline,image=str(path),sha256=sha(path)))
    return image,meta

def examples(output,library):
    """No model initialization: replay genuine cached words with exact original seeds."""
    from ..layout import build
    directory=ROOT/output
    if (directory/'report.json').exists():raise FileExistsError('Use a fresh --output directory to preserve previous examples')
    directory.mkdir(parents=True,exist_ok=True)
    equation_results=[];tiles=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for n,expression in enumerate(EXAMPLES,1):
        image,meta=save_equation(expression,str(Path(output)/f'equation_{n}.png'),library,9600+n)
        tile=Image.new('RGB',(760,max(115,image.height+65)),'white');draw=ImageDraw.Draw(tile)
        label='APPROVED HANDWRITTEN SYMBOL CROPS' if meta['source']=='handwritten_math_symbols' else 'STANDARD FONT FALLBACK — NOT HANDWRITING'
        draw.text((12,8),f'{n}. {expression} | {label}',font=font,fill=(50,50,50));tile.paste(image,(22,45))
        tiles.append(tile);equation_results.append(dict(meta,output=str(Path(output)/f'equation_{n}.png')))
    sheet=Image.new('RGB',(760,sum(tile.height for tile in tiles)),'white');y=0
    for tile in tiles:sheet.paste(tile,(0,y));y+=tile.height
    sheet.save(directory/'equations_contact.png')
    original_path=ROOT/'outputs/quality_tests/after_final.json'
    original=read_json(original_path)
    protected={str(p.relative_to(ROOT)):sha(p) for folder in ['data/pages','outputs/evaluation','outputs/finetune','outputs/quality_tests']
               for p in (ROOT/folder).rglob('*') if p.is_file()}
    baseline=ROOT/'outputs/erdem_test.png'
    if baseline.exists():protected[str(baseline.relative_to(ROOT))]=sha(baseline)
    class Replay:
        class Pipeline:identity=original['model']
        pipeline=Pipeline();settings=original['generation'];hits=generated=0
        def obtain(self,text,seed):
            record=next((r for r in original['word_records'] if r['text']==text and r['seed']==seed),None)
            if record is None:raise ValueError('No matching cached generated word: '+text)
            self.hits+=1
            path=ROOT/'.reforge-cache/words'/f'{record["cache_id"]}.png'
            with Image.open(path) as image:return image.convert('RGBA'),dict(record,id=record['cache_id'])
    # Keep prose unchanged so no expensive inference or approvals are fabricated.
    doc=dict(original['document'])
    doc['blocks']=original['document']['blocks'][:3]+[{'kind':'math','text':text} for text in EXAMPLES]
    doc['layout']=dict(doc['layout'],x_height_mm=3.1,size_variation=.025)
    doc['math']={'library':str(library.directory),'mode':'auto','seed':9600,'variation':.025,'size_scale':1}
    page=build(doc,Replay(),str(Path(output)/'physics'),original['profile'],original['overrides'])
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Protected file changed: '+name)
    paths=[directory/'equations_contact.png']+[directory/f'equation_{i}.png' for i in range(1,7)]+[ROOT/p for p in page['pages']]
    sizes={}
    for path in paths:
        with Image.open(path) as image:image.load();sizes[str(path.relative_to(ROOT))]=list(image.size)
    report={'step':'9.6','symbol_library':library.identity(),'equations':equation_results,'pages':page['pages'],
            'image_readability_verified':sizes,'protected_files_verified':len(protected),'pretrained_model':original['model'],
            'prose_source_manifest':str(original_path.relative_to(ROOT)),'prose_inference_performed':False,
            'prose_cache_hits':page['cache_hits'],'math_missing_symbols':sorted({s for e in equation_results for s in e.get('missing_symbols',[])}),
            'limitations':['Only explicitly user-reviewed symbol crops are eligible. No mathematical labels were inferred from scans.',
                           'If approved symbols are absent, examples intentionally use labeled standard-font fallback; they do not demonstrate personalized handwriting yet.',
                           'Prose reuses existing actual pretrained generations, still pending human review; malformed model words remain possible.',
                           'Rectangular cropping cannot reliably remove overlapping grid lines or isolate touching symbols. Choose a clean occurrence and review the ink cutoff.',
                           'Symbol assembly preserves mathematical layout but does not reproduce pen connections between separate glyphs.']}
    write_json(directory/'report.json',report)
    print('Saved:',page['pages']);print('Approved crops:',library.identity()['approved_entries']);print('Missing:',report['math_missing_symbols'])

def main():
    # Windows redirected consoles can use cp1252; Greek inventory logging must not abort.
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(errors='backslashreplace')
    parser=argparse.ArgumentParser(description='Collect explicitly reviewed symbols and compose restricted mathematical expressions')
    parser.add_argument('--library',default=DEFAULT_LIBRARY)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('collect')
    single=sub.add_parser('render');single.add_argument('--expression',required=True);single.add_argument('--output',required=True);single.add_argument('--seed',type=int,default=9600)
    demo=sub.add_parser('examples');demo.add_argument('--output',default='outputs/math_tests')
    args=parser.parse_args();library=Library(args.library)
    if args.command=='collect':
        from .collector import launch
        launch(library)
    elif args.command=='render':
        _,meta=save_equation(args.expression,args.output,library,args.seed);print(meta['source'],'missing:',meta.get('missing_symbols',[]))
    else:examples(args.output,library)

if __name__=='__main__':main()
