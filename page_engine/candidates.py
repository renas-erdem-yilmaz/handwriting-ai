"""Persistent seed candidates and explicit manual decisions; OCR stays optional."""
import hashlib
import json
import shutil
from PIL import Image
from .common import ROOT, read_json, write_json, sha
from .words import WordStore

class CandidateStore:
    def __init__(self, pipeline, settings, directory=None, initial_count=3, reuse_approved=False):
        if not 1 <= initial_count <= 3: raise ValueError('Initial candidate count must be 1–3')
        self.pipeline = pipeline
        self.settings = dict(settings, retries=0)  # Save each distinct seed; no hidden best-only retry.
        self.raw = WordStore(pipeline, self.settings)
        self.directory = directory or ROOT / '.reforge-cache/candidate_reviews'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.initial_count, self.reuse_approved = initial_count, reuse_approved
        self.hits = self.generated = 0
        self.legacy = None
        self.processing = {'mode': 'adaptive', 'opacity_target': pipeline.style_config.get('ink_opacity_target', .82)}

    def key(self, text, seed):
        request = {'text': text, 'seed': seed, 'sampling': self.settings, 'model': self.pipeline.identity}
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24], request

    def path(self, bundle_id):
        if len(bundle_id) != 24 or any(c not in '0123456789abcdef' for c in bundle_id): raise ValueError('Invalid bundle ID')
        return self.directory / (bundle_id + '.json')

    def load(self, bundle_id): return read_json(self.path(bundle_id))

    def import_legacy(self, text, seed):
        """Reuse an exact saved seed only if it was not substituted by a retry."""
        if self.legacy is None:
            self.legacy = {}
            for path in self.raw.directory.glob('*.json'):
                record = read_json(path); req = record['request']; sampling = req['settings']
                if req['model'] == self.pipeline.identity and sampling['steps'] == self.settings['steps'] and sampling['guidance'] == self.settings['guidance'] and not sampling['ocr'] and not self.settings['ocr']:
                    selected = record['attempts'][record['selected_attempt']]
                    if selected['seed'] == req['seed'] and record['status'] != 'rejected':
                        self.legacy[(req['text'], req['seed'])] = record
        old = self.legacy.get((text, seed))
        if not old: return
        request = {'text': text, 'seed': seed, 'settings': self.settings, 'model': self.pipeline.identity}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        path = self.raw.directory / (key + '.png')
        if path.with_suffix('.json').exists(): return
        source = ROOT / old['image']
        if sha(source) != old['image_sha256']: raise ValueError('Legacy image hash mismatch')
        shutil.copyfile(source, path)
        record = dict(old, id=key, request=request, image=str(path.relative_to(ROOT)),
                      attempts=[old['attempts'][old['selected_attempt']]], selected_attempt=0, imported_from=old['id'])
        write_json(path.with_suffix('.json'), record)

    def ensure(self, text, seed):
        key, request = self.key(text, seed)
        path = self.path(key)
        if path.exists():
            bundle = read_json(path)
            if bundle['request'] != request: raise ValueError('Review request mismatch')
        else:
            bundle = {'id': key, 'request': request, 'candidates': [], 'selected': None,
                      'status': 'pending', 'problem_note': '', 'decision_history': []}
            if self.reuse_approved:
                for prior_path in sorted(self.directory.glob('*.json')):
                    prior = read_json(prior_path)
                    if prior['status'] == 'approved' and prior['request']['text'] == text and prior['request']['sampling'] == self.settings and prior['request']['model'] == self.pipeline.identity:
                        bundle.update(candidates=prior['candidates'], selected=prior['selected'], status='approved',
                                      approval_inherited_from=prior['id'], problem_note='Reused previously approved exact word candidate')
                        write_json(path,bundle);break
        while len(bundle['candidates']) < self.initial_count:
            self.add_to(bundle)
        return bundle

    def add_to(self, bundle, seed=None):
        # Regeneration appends, preserving prior candidates and decisions.
        occupied = {c['seed'] for c in bundle['candidates']}
        if seed is None:
            seed = bundle['request']['seed'] + len(bundle['candidates']) * 104729
            while seed in occupied: seed += 104729
        if seed in occupied: raise ValueError('Choose a seed not already in this word bundle')
        if not isinstance(seed,int) or not 0<=seed<2**63:raise ValueError('Seed must be an integer between 0 and 2^63-1')
        text = bundle['request']['text']
        self.import_legacy(text, seed)
        before = self.raw.generated
        image, record = self.raw.obtain(text, seed)
        self.generated += self.raw.generated - before
        if self.raw.generated == before: self.hits += 1
        bundle['candidates'].append({'seed': seed, 'cache_id': record['id'], 'image': record['image'],
                                     'sha256': record['image_sha256'], 'flags': record['flags'], 'ocr': record['attempts'][0]['ocr']})
        write_json(self.path(bundle['id']), bundle)
        return bundle

    def append(self, bundle_id, seed=None):
        bundle = self.load(bundle_id)
        if bundle['request']['model'] != self.pipeline.identity or bundle['request']['sampling'] != self.settings:
            raise ValueError('Cannot regenerate with different model, references or sampling')
        return self.add_to(bundle, seed)

    def decide(self, bundle_id, selected=None, status='approved', note=''):
        if status not in ('approved', 'pending', 'problematic'): raise ValueError('Invalid review status')
        bundle = self.load(bundle_id)
        if selected is not None:
            if not 0 <= selected < len(bundle['candidates']): raise ValueError('Candidate index outside bundle')
            bundle['selected'] = selected
        if status == 'approved' and bundle['selected'] is None: raise ValueError('Select a candidate before approval')
        bundle['status'], bundle['problem_note'] = status, note
        bundle['decision_history'].append({'selected': bundle['selected'], 'status': status, 'note': note})
        write_json(self.path(bundle_id), bundle)
        return bundle

    def obtain(self, text, seed):
        bundle = self.ensure(text, seed)
        selected = bundle['selected'] if bundle['selected'] is not None else 0
        chosen = bundle['candidates'][selected]
        path = ROOT / chosen['image']
        if sha(path) != chosen['sha256']: raise ValueError('Candidate image hash mismatch')
        if bundle['status'] == 'problematic':
            flags = [*chosen['flags'], 'manually_problematic']
        else: flags = chosen['flags']
        return Image.open(path).convert('L'), {'id': chosen['cache_id'], 'status': bundle['status'], 'flags': flags,
                                               'bundle_id': bundle['id'], 'candidate_index': selected}
