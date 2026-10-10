"""Lazy, strictly checked loading of the existing local pretrained pipeline."""
import os
import random
import sys
import numpy as np
from PIL import Image
from .common import ROOT, read_json, sha

def prepare_imports():
    cfg = read_json(ROOT / 'training/config.json')
    os.environ['HF_HOME'] = str(ROOT / cfg['hf_cache'])
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT / 'reforge'))
    return cfg

class Pipeline:
    def __init__(self, adapter=None, device='auto', style_config=None):
        self.cfg = prepare_imports()
        self.adapter = ROOT / adapter if adapter else None
        self.style_config_path = style_config or 'page_engine/styles/erdem.json'
        self.style_config = read_json(ROOT / self.style_config_path)
        self.refs = [ROOT / p for p in self.style_config['references']]
        if len(self.refs) != 5 or len(set(self.refs)) != 5:
            raise ValueError('Style configuration must select five distinct word crops')
        for path in self.refs:
            with Image.open(path) as image:
                image.verify()
        cfg = self.cfg
        snapshot = ROOT / cfg['hf_cache'] / 'hub' / ('models--' + cfg['checkpoint_repo'].replace('/', '--')) / 'snapshots' / cfg['checkpoint_revision']
        self.base_path = snapshot / cfg['unet_checkpoint']
        self.style_path = snapshot / cfg['style_checkpoint']
        self.identity = {'base_sha256': sha(self.base_path), 'style_encoder_sha256': sha(self.style_path),
                         'adapter_sha256': sha(self.adapter) if self.adapter else None,
                         'references': {str(p.relative_to(ROOT)): sha(p) for p in self.refs},
                         'text_revision': cfg['text_revision'], 'vae_revision': cfg['vae_revision'],
                         'implementation': 'step9-v1'}
        self.device_option = device
        self.ready = False

    def load(self):
        if self.ready: return
        import torch
        from reforge.model import weights, encoder, generator
        from reforge.preprocess.normalize import preprocess_words
        # Resolve auxiliary model revisions explicitly, never a floating HF main.
        hub = ROOT / self.cfg['hf_cache'] / 'hub'
        weights.CANINE_MODEL_ID = str(hub / ('models--' + self.cfg['text_model'].replace('/', '--')) / 'snapshots' / self.cfg['text_revision'])
        weights.SD_MODEL_ID = str(hub / ('models--' + self.cfg['vae_model'].replace('/', '--')) / 'snapshots' / self.cfg['vae_revision'])
        torch.set_num_threads(4)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device_option == 'auto' else self.device_option
        torch.backends.cuda.matmul.allow_tf32 = True
        # Reforge constructs the exact architecture; re-load all keys strictly.
        self.unet = weights.load_unet(str(self.base_path), device=self.device)
        state = weights.strip_module_prefix(torch.load(self.base_path, map_location='cpu', weights_only=True))
        self.unet.load_state_dict(state, strict=True)
        if self.adapter:
            checkpoint = torch.load(self.adapter, map_location='cpu', weights_only=True)
            if checkpoint['config'] != self.cfg:
                raise ValueError('Adapter training configuration does not match the pinned base')
            selected = checkpoint['adapter']
            if set(selected) != {'style_lin.weight', 'style_lin.bias'}:
                raise ValueError('Unexpected adapter keys')
            for key, value in selected.items():
                if value.shape != state[key].shape or value.dtype != state[key].dtype or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter tensor: ' + key)
            self.unet.style_lin.load_state_dict({k.split('.', 1)[1]: v for k, v in selected.items()}, strict=True)
        del state
        self.encoder = encoder.StyleEncoder(None)
        s = weights.strip_module_prefix(torch.load(self.style_path, map_location='cpu', weights_only=True))
        # These two triplet-training classifier tensors are explicitly unused.
        for key in ['classifier.weight', 'classifier.bias']: s.pop(key, None)
        self.encoder.encoder.load_state_dict(s, strict=True)
        self.encoder.to(self.device).requires_grad_(False).eval()
        self.raw_refs = [np.asarray(Image.open(p).convert('L')) for p in self.refs]
        self.features = self.encoder.encode(preprocess_words(self.raw_refs))
        self.vae = weights.load_vae(device=self.device).requires_grad_(False).eval()
        self.tokenizer = weights.load_tokenizer()
        self.unet.requires_grad_(False).eval()
        self.generator = generator
        self.ready = True
        print('Strict pretrained checkpoints loaded; device=' + self.device, flush=True)

    def word(self, text, seed, steps, guidance):
        self.load()
        import torch
        random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        context = self.tokenizer(' ', return_tensors='pt', padding='max_length', max_length=16)
        # Reforge's implicit OCR retries are disabled. Our explicit validator is optional.
        # Single-threaded inference: do not share this pipeline across concurrent threads.
        original = self.generator._get_ocr_fn
        cwd = os.getcwd()
        try:
            self.generator._get_ocr_fn = lambda: None
            os.chdir(ROOT / 'reforge')  # Repository punctuation font uses relative paths.
            with torch.inference_mode():
                result = self.generator.generate_word(text, self.unet, self.vae, self.tokenizer, self.features,
                    uncond_context=context, num_steps=steps, guidance_scale=guidance, num_candidates=1,
                    device=self.device, style_reference_imgs=self.raw_refs)
        finally:
            self.generator._get_ocr_fn = original
            os.chdir(cwd)
        if result is None: raise RuntimeError('Generator returned no word: ' + text)
        return Image.fromarray(np.asarray(result).astype('uint8')).convert('L')
