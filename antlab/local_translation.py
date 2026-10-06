"""Optional local M2M100 translator. Inference never downloads or sends text."""
from pathlib import Path
import sys
import threading
import pickle
from .translation_gateway import TranslationError

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'facebook/m2m100_418M'
# Model card's supported ISO language codes, not a claim of measured quality.
LANGUAGES = tuple('af am ar ast az ba be bg bn br bs ca ceb cs cy da de el en es et fa ff fi fr fy ga gd gl gu ha he hi hr ht hu hy id ig ilo is it ja jv ka kk km kn ko lb lg ln lo lt lv mg mk ml mn mr ms my ne nl no ns oc or pa pl ps pt ro ru sd si sk sl so sq sr ss su sv sw ta th tl tn tr uk ur uz vi wo xh yi yo zh zu'.split())


def optional_dependencies():
    directory = ROOT / '.translation-deps'
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


class M2M100Translator:
    name = 'local-m2m100-418M'

    def __init__(self, model_directory, threads=2, max_input_tokens=256, max_new_tokens=128):
        if any(type(v) is not int or v < 1 for v in (threads, max_input_tokens, max_new_tokens)):
            raise ValueError('positive integer runtime limits required')
        self.directory = Path(model_directory)
        self.threads = threads
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._model = self._tokenizer = None
        self._lock = threading.Lock()

    def supports(self, source_lang, target_lang):
        return source_lang in LANGUAGES and target_lang in LANGUAGES

    def _load(self):
        if self._model is not None:
            return
        if not all((self.directory / name).is_file() for name in ('config.json', 'vocab.json', 'sentencepiece.bpe.model', 'pytorch_model.bin')):
            raise TranslationError('local translation model unavailable')
        optional_dependencies()
        try:
            import torch
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError:
            raise TranslationError('optional translation dependencies unavailable') from None
        torch.set_num_threads(self.threads)
        try:
            tokenizer = M2M100Tokenizer.from_pretrained(self.directory, local_files_only=True)
            model = M2M100ForConditionalGeneration.from_pretrained(
                self.directory, local_files_only=True, use_safetensors=False,
                weights_only=True, torch_dtype=torch.float32).eval()
            if set(tokenizer.lang_code_to_id) != set(LANGUAGES):
                raise TranslationError('translation vocabulary differs from declared languages')
        except (OSError, ValueError, pickle.UnpicklingError, TranslationError):
            self._model = self._tokenizer = None
            raise TranslationError('local translation model unavailable or invalid') from None
        self._tokenizer, self._model = tokenizer, model

    def translate(self, text, source_lang, target_lang):
        if not isinstance(text, str) or not text.strip() or not self.supports(source_lang, target_lang):
            raise TranslationError('unsupported language or empty source')
        if source_lang == target_lang:
            return text
        # Tokenizer source language is mutable; serialize calls through this
        # backend instance instead of mixing language settings across threads.
        with self._lock:
            self._load()
            import torch
            self._tokenizer.src_lang = source_lang
            tokens = self._tokenizer(text, return_tensors='pt', truncation=False)
            if tokens['input_ids'].shape[1] > self.max_input_tokens:
                raise TranslationError('source exceeds token limit; no truncation performed')
            with torch.inference_mode():
                output = self._model.generate(**tokens,
                    forced_bos_token_id=self._tokenizer.get_lang_id(target_lang),
                    max_new_tokens=self.max_new_tokens, num_beams=3, do_sample=False)
            if output.shape[1] >= self.max_new_tokens + 1:
                raise TranslationError('translation reached output limit')
            translated = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if not translated.strip():
                raise TranslationError('empty translation')
            return translated
