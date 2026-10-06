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

    def __init__(self, model_directory, threads=2, max_input_tokens=256, max_new_tokens=128, batch_size=4):
        if any(type(v) is not int or v < 1 for v in (threads, max_input_tokens, max_new_tokens, batch_size)):
            raise ValueError('positive integer runtime limits required')
        self.directory = Path(model_directory)
        self.threads = threads
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
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
        except (OSError, ValueError, pickle.UnpicklingError, TranslationError, torch.OutOfMemoryError):
            self._model = self._tokenizer = None
            raise TranslationError('local translation model unavailable or invalid') from None
        self._tokenizer, self._model = tokenizer, model

    def translate(self, text, source_lang, target_lang):
        if not isinstance(text, str) or not text.strip() or not self.supports(source_lang, target_lang):
            raise TranslationError('unsupported language or empty source')
        translated = self.translate_many([text], source_lang, target_lang)[0]
        if translated is None:
            raise TranslationError('translation unavailable or exceeds limits')
        return translated

    def translate_many(self, texts, source_lang, target_lang):
        """Bounded local batches; failed rows are None, without losing ordering.

        Empty/over-limit rows never enter model generation. Identical segments
        reuse a result within this invocation only, with no persistent cache.
        """
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise ValueError('supply a list of source strings')
        if not self.supports(source_lang, target_lang):
            raise TranslationError('unsupported language')
        if not texts:
            return []
        if source_lang == target_lang:
            return [text if text.strip() else None for text in texts]
        # Tokenizer source language is mutable; serialize calls through this
        # backend instance instead of mixing language settings across threads.
        with self._lock:
            self._load()
            import torch
            self._tokenizer.src_lang = source_lang
            unique = list(dict.fromkeys(texts))
            results = dict.fromkeys(unique)
            nonempty = [text for text in unique if text.strip()]
            if not nonempty:
                return [None] * len(texts)
            for start in range(0, len(nonempty), self.batch_size):
                batch = nonempty[start:start+self.batch_size]
                encoded = self._tokenizer(batch, truncation=False)
                chunk = [(text, {key: value[index] for key, value in encoded.items()})
                         for index, text in enumerate(batch)
                         if len(encoded['input_ids'][index]) <= self.max_input_tokens]
                if not chunk:
                    continue
                try:
                    tokens = self._tokenizer.pad([row for _, row in chunk], padding=True, return_tensors='pt')
                    with torch.inference_mode():
                        output = self._model.generate(**tokens,
                            forced_bos_token_id=self._tokenizer.get_lang_id(target_lang),
                            max_new_tokens=self.max_new_tokens, num_beams=3, do_sample=False)
                except torch.OutOfMemoryError:
                    # Keep these rows unavailable and retain every original;
                    # unrelated chunks may still translate successfully.
                    continue
                decoded = self._tokenizer.batch_decode(output, skip_special_tokens=True)
                for (text, _), row, translated in zip(chunk, output.tolist(), decoded):
                    # Ignore batch padding, but reject a row reaching the
                    # generation limit even if a forced EOS ends that row.
                    generated = [token for token in row[1:] if token != self._tokenizer.pad_token_id]
                    if len(generated) < self.max_new_tokens and translated.strip():
                        results[text] = translated
            return [results[text] for text in texts]
