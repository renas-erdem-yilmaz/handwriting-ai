"""Immutable source crops; only explicitly reviewed entries are renderable."""
from datetime import datetime, timezone
from pathlib import Path
import uuid
import time
import numpy as np
from PIL import Image
from ..common import ROOT, read_json, write_json, sha

DEFAULT_LIBRARY = '.reforge-cache/math_symbols'
LABELS = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZαβθφΔωγπμσλ+-−=×÷⇒→≤≥≠±()[].,') + ['fraction_bar']

def prepared(raw, cutoff=220):
    """Conservative white-paper removal, no morphology or shape changes."""
    rgba = raw.convert('RGBA')
    white = Image.new('RGBA', rgba.size, 'white'); white.alpha_composite(rgba)
    gray = np.asarray(white.convert('L'))
    mask = gray < cutoff
    ys, xs = np.where(mask)
    if not len(xs): raise ValueError('No visible ink at this cutoff; adjust it or select another crop')
    box = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
    # Keep dark pixels' original luminance. Pale paper/grid is removed only on a copy.
    clean = np.where(mask, gray, 255).astype('uint8')
    return Image.fromarray(clean).crop(box), box

class Library:
    def __init__(self, directory=DEFAULT_LIBRARY):
        self.directory = ROOT / directory
        self.path = self.directory / 'library.json'
        self.data = read_json(self.path) if self.path.exists() else {'schema':1, 'entries':[]}
        if self.data.get('schema') != 1: raise ValueError('Unsupported symbol library schema')

    def save(self):
        # OneDrive/Windows can briefly lock the destination of an atomic rename.
        for attempt in range(5):
            try:write_json(self.path,self.data);return
            except PermissionError:
                if attempt==4:raise
                time.sleep(.05*(attempt+1))

    def capture(self, source, rect, label, cutoff=220, baseline=.85, height_em=1.0):
        self.validate(label, cutoff, baseline, height_em)
        source = Path(source).resolve()
        with Image.open(source) as page:
            x,y,r,b = map(int,rect)
            if not (0<=x<r<=page.width and 0<=y<b<=page.height): raise ValueError('Crop outside page')
            raw = page.crop((x,y,r,b)).convert('RGBA')
        prepared(raw, cutoff)  # Blank crops cannot be approved accidentally.
        entry_id = uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)
        crop_path = self.directory / (entry_id + '.png')
        raw.save(crop_path)  # Never edit the source page.
        entry = {'id':entry_id, 'label':label, 'image':crop_path.name, 'sha256':sha(crop_path),
                 'source_page':str(source), 'source_sha256':sha(source), 'rect':[x,y,r,b],
                 'cutoff':cutoff, 'baseline':baseline, 'height_em':height_em,
                 'status':'pending', 'review':None, 'history':[]}
        self.data['entries'].append(entry); self.save()
        return entry

    @staticmethod
    def validate(label, cutoff, baseline, height_em):
        if label not in LABELS: raise ValueError('Use a single supported symbol or fraction_bar')
        if not 80<=cutoff<=250 or not 0<=baseline<=1 or not .08<=height_em<=3:
            raise ValueError('Cutoff: 80–250; baseline: 0–1; height: 0.08–3 x-height units')

    def update(self, entry_id, label, cutoff, baseline, height_em, approve=False, reviewed=False):
        self.validate(label,cutoff,baseline,height_em)
        if approve and not reviewed: raise ValueError('Explicit crop AND label review is required')
        entry = self.get(entry_id)
        if sha(self.directory / entry['image']) != entry['sha256']: raise ValueError('Raw crop changed; capture a new variant')
        with Image.open(self.directory / entry['image']) as raw: prepared(raw,cutoff)
        entry['history'].append({k:entry[k] for k in ('label','cutoff','baseline','height_em','status','review')})
        entry.update(label=label,cutoff=cutoff,baseline=baseline,height_em=height_em,
                     status='approved' if approve else 'pending',
                     review={'confirmed_crop_and_label':True,'time_utc':datetime.now(timezone.utc).isoformat()} if approve else None)
        self.save(); return entry

    def get(self, entry_id): return next(e for e in self.data['entries'] if e['id']==entry_id)

    def variants(self, label):
        return [e for e in self.data['entries'] if e['label']==label and e['status']=='approved'
                and (e.get('review') or {}).get('confirmed_crop_and_label') is True]

    def image(self, entry):
        path = self.directory / entry['image']
        if path.resolve().parent != self.directory.resolve(): raise ValueError('Invalid crop path')
        if sha(path) != entry['sha256']: raise ValueError('Approved crop changed: '+entry['id'])
        with Image.open(path) as raw: return prepared(raw, entry['cutoff'])[0]

    def identity(self):
        return {'path':str(self.directory), 'sha256':sha(self.path) if self.path.exists() else None,
                'approved_entries':sum(len(self.variants(label)) for label in set(LABELS))}
