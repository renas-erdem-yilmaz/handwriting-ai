import hashlib
import json
import re
import time
from pathlib import Path
from PIL import Image
from .common import ROOT, read_json, write_json, sha
from .normalize import assess

class WordStore:
    def __init__(self, pipeline, settings):
        self.pipeline, self.settings = pipeline, settings
        self.directory = ROOT / '.reforge-cache/words'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = self.generated = 0

    def obtain(self, text, seed):
        if not re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*[.,;:!?]?", text):
            raise ValueError(f'Unsupported handwriting token {text!r}; use explicit $math$ notation')
        request = {'text': text, 'seed': seed, 'settings': self.settings, 'model': self.pipeline.identity}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        path = self.directory / (key + '.png'); meta = path.with_suffix('.json')
        if meta.exists():
            record = read_json(meta)
            if record['request'] != request or sha(path) != record['image_sha256']:
                raise ValueError('Corrupt cache entry: ' + key)
            if record['status'] == 'rejected': raise ValueError('Manually rejected word; supply a replacement seed: ' + key)
            self.hits += 1
            return Image.open(path).convert('L'), record
        attempts = []
        best = None
        for i in range(self.settings['retries'] + 1):
            actual_seed = seed + i * 104729
            start = time.monotonic()
            image = self.pipeline.word(text, actual_seed, self.settings['steps'], self.settings['guidance'])
            flags, penalty = assess(image, text)
            recognized = None
            if self.settings['ocr']:
                from reforge.evaluate.ocr import ocr_read
                import numpy as np
                recognized = ocr_read(np.asarray(image))
                norm = lambda s: re.sub('[^a-z]', '', s.lower())
                if norm(recognized) != norm(text): flags.append('ocr_mismatch'); penalty += 2
            attempts.append({'seed': actual_seed, 'flags': flags, 'ocr': recognized, 'seconds': time.monotonic() - start})
            if best is None or penalty < best[0]: best = (penalty, image, i)
            if not flags: break
        image = best[1]
        image.save(path)
        record = {'id': key, 'request': request, 'image': str(path.relative_to(ROOT)), 'image_sha256': sha(path),
                  'attempts': attempts, 'selected_attempt': best[2], 'flags': attempts[best[2]]['flags'],
                  'status': 'pending', 'spelling_verified': False,
                  'note': 'Structural checks/retries cannot guarantee spelling; manual approval is separate.'}
        write_json(meta, record); self.generated += 1
        print(f'Word {key}: {text} seed={seed} flags={record["flags"]}', flush=True)
        return image, record

def review(word_id, status):
    if not re.fullmatch('[a-f0-9]{24}', word_id): raise ValueError('Invalid word ID')
    path = ROOT / '.reforge-cache/words' / (word_id + '.json')
    record = read_json(path)
    record['status'] = status
    write_json(path, record)
    return record
