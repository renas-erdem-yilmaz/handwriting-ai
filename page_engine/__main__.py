import argparse
from .common import ROOT, read_json, write_json
from .model import Pipeline, prepare_imports
from .words import WordStore, review
from .layout import build
from .profile import analyze

def main():
    p = argparse.ArgumentParser(description='Step 9 local handwriting preview tools; never trains.')
    sub = p.add_subparsers(dest='command', required=True)
    profile = sub.add_parser('analyze'); profile.add_argument('--output', default='page_engine/profiles/scanned.json')
    render = sub.add_parser('render')
    render.add_argument('--document', required=True); render.add_argument('--output', required=True)
    render.add_argument('--profile', default='page_engine/profiles/scanned.json')
    render.add_argument('--adapter'); render.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    render.add_argument('--steps', type=int, default=20); render.add_argument('--guidance', type=float, default=3.0)
    render.add_argument('--retries', type=int, default=1); render.add_argument('--ocr', action='store_true')
    render.add_argument('--overrides', help='JSON object mapping word occurrence index to replacement seed')
    render.add_argument('--require-approved', action='store_true')
    approve = sub.add_parser('review'); approve.add_argument('word_id'); approve.add_argument('status', choices=['approved', 'rejected', 'pending'])
    regen = sub.add_parser('regenerate'); regen.add_argument('--manifest', required=True); regen.add_argument('--word-index', type=int, required=True)
    regen.add_argument('--seed', type=int, required=True); regen.add_argument('--output', required=True)
    sheet = sub.add_parser('contact-sheet'); sheet.add_argument('--manifest', required=True); sheet.add_argument('--output', required=True)
    args = p.parse_args()
    if args.command == 'analyze':
        print('Measured pages:', analyze(ROOT / args.output)['page_count']); return
    if args.command == 'review':
        print(review(args.word_id, args.status)['status']); return
    if args.command == 'contact-sheet':
        from PIL import Image, ImageDraw, ImageFont
        manifest = read_json(ROOT / args.manifest)
        records = manifest['word_records']; width = 1000; row_height = 100
        image = Image.new('RGB', (width, max(1, len(records)) * row_height), 'white'); draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
        for row, item in enumerate(records):
            record = read_json(ROOT / '.reforge-cache/words' / (item['cache_id'] + '.json'))
            word = Image.open(ROOT / record['image']).convert('RGB'); word.thumbnail((420, 72))
            image.paste(word, (10, row * row_height + 24))
            draw.text((450, row * row_height + 8), f'Index {item["index"]}: {item["text"]} | {record["status"]}', font=font, fill='black')
            draw.text((450, row * row_height + 34), item['cache_id'], font=font, fill='black')
            draw.text((450, row * row_height + 58), ', '.join(record['flags']) or 'No structural flag; spelling unverified', font=font, fill='black')
        target = ROOT / args.output
        if target.exists(): raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True); image.save(target); return
    prepare_imports()
    if args.command == 'regenerate':
        previous = read_json(ROOT / args.manifest)
        found = next(r for r in previous['word_records'] if r['index'] == args.word_index)
        if args.seed == found['seed']: raise ValueError('Choose a different seed to regenerate')
        adapter_hash = previous['model']['adapter_sha256']
        adapter = None
        if adapter_hash:
            raise ValueError('For an adapter preview use render --adapter with --overrides; explicit checkpoint path required')
        overrides = dict(previous['overrides']); overrides[str(args.word_index)] = args.seed
        pipeline = Pipeline(adapter)
        if pipeline.identity != previous['model']: raise ValueError('Model/reference changed since original preview')
        store = WordStore(pipeline, previous['generation'])
        result = build(previous['document'], store, args.output, previous['profile'], overrides)
    else:
        if not 1 <= args.steps <= 200 or not 0 <= args.retries <= 10 or not 0 < args.guidance <= 20: raise ValueError('Unsafe sampling bounds')
        settings = {'steps': args.steps, 'guidance': args.guidance, 'retries': args.retries, 'ocr': args.ocr}
        pipeline = Pipeline(args.adapter, args.device); store = WordStore(pipeline, settings)
        result = build(read_json(ROOT / args.document), store, args.output, read_json(ROOT / args.profile),
                       read_json(ROOT / args.overrides) if args.overrides else None, args.require_approved)
    print('Pages:', result['pages'], 'cache hits:', result['cache_hits'], 'new words:', result['new_words'])

if __name__ == '__main__': main()
