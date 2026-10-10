"""Recompose exactly the saved raw words; no inference or approval changes."""
from PIL import Image
from .common import ROOT,read_json,write_json,sha
from .layout import build
from .quality_experiment import panels

def main():
    original=read_json(ROOT/'outputs/quality_tests/after_final.json')
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'outputs/quality_tests').glob('*') if p.is_file()}
    class Replay:
        class Pipeline:identity=original['model']
        pipeline=Pipeline();settings=original['generation'];hits=generated=0
        def obtain(self,text,seed):
            record=next(r for r in original['word_records'] if r['text']==text and r['seed']==seed)
            self.hits+=1
            return Image.open(ROOT/'.reforge-cache/words'/f'{record["cache_id"]}.png').convert('RGBA'),dict(record,id=record['cache_id'])
    document=dict(original['document'])
    document['layout']=dict(document['layout'],x_height_mm=3.1,size_variation=.05,heading_size_scale=1.25)
    prefix='outputs/size_tests/consistent_final'
    result=build(document,Replay(),prefix,original['profile'],original['overrides'])
    assert [w['cache_id'] for w in result['word_records']]==[w['cache_id'] for w in original['word_records']]
    panels([original['pages'][0],result['pages'][0]],['BEFORE: total-height / proxy scaling','AFTER: shared x-height, bounded variation'],ROOT/'outputs/size_tests/comparison_final.png')
    for name,digest in protected.items():assert sha(ROOT/name)==digest,name
    diagnostics=[{'word':p['text'],**p['rendering']} for p in result['positions'] if p['rendering']['source']=='diffusionpen_word']
    write_json(ROOT/'outputs/size_tests/measurements_final.json',{'same_cached_words':True,'inference_performed':False,
        'previous_quality_outputs_unchanged':True,'writing_scale':result['writing_scale'],'words':diagnostics,
        'limits':['Ink-derived x-height estimates are heuristic; fallback words may retain body-size differences.',
                  'Malformed letters and inconsistent pen geometry originate in the model, not resizing.',
                  'Heading size is intentionally 1.25x body size; configure heading_size_scale=1 for uniform titles.']})
    print('Saved:',result['pages'])

if __name__=='__main__':main()
