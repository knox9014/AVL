"""Explicit-language English pivot around the frozen semantic vector channel.

Translation is a separately resourced outer layer, not part of the trained
99,977-parameter model or evidence of multilingual semantic retention. This
module calls an injected backend only; it contains no remote service, automatic
language detector, semantic keyword canonicalizer or hidden label lookup.
Sources and translations are returned locally for audit/fallback, never placed
in the vector packet or supplied to the receiver. A backend's translation can
be wrong even when its English result is admitted by the finite grammar.
"""
import time
from typing import Protocol

import torch

from .semantic_codec import pack_vectors
from .semantic_render import render_meaning
from .semantic_v2_data import is_supported
from .semantic_v2_demo import receive_packet
from .semantic_v2_model import tokenize


class TranslationError(Exception):
    """Expected local backend unavailability; details are never returned."""


class Translator(Protocol):
    name: str

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """Whether this configured backend supports the explicit direction."""
        ...

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Return a translation, or raise TranslationError if unavailable."""
        ...


def _language(language):
    if not isinstance(language, str) or not language or language != language.strip():
        raise ValueError('supply an explicit nonempty language identifier')
    return language


class TranslationGateway:
    """Translate locally at the edges; only codec bytes cross the receiver boundary.

    Backends must use TranslationError for an expected model/resource failure.
    Unexpected programming errors propagate instead of becoming success-shaped
    fallback output. Returned errors never include exception messages or paths.
    """

    def __init__(self, model, translator: Translator | None = None):
        self.model = model
        self.translator = translator
        if translator is not None and (not isinstance(translator.name, str) or not translator.name):
            raise ValueError('translator must supply a nonempty backend name')

    def _metadata(self):
        return dict(backend=self.translator.name if self.translator is not None else None,
                    translation_calls=0, translation_seconds=0., model_seconds=0.,
                    translation_input_characters=0, translation_output_characters=0,
                    cost=dict(amount=None, currency=None, measured=False,
                              scope='translation backend resources are separate from the trained semantic model'),
                    scope='English-pivot interface; multilingual semantic accuracy not validated',
                    originals_retained_locally=True)

    def _translate(self, text, source_lang, target_lang, metadata):
        if self.translator is None:
            return None, 'translation_unavailable'
        started = time.perf_counter()
        try:
            try:
                supported = self.translator.supports(source_lang, target_lang)
                if not isinstance(supported, bool):
                    raise TypeError('translator supports must return a boolean')
                if not supported:
                    return None, 'unsupported_language'
                metadata['translation_calls'] += 1
                metadata['translation_input_characters'] += len(text)
                translated = self.translator.translate(text, source_lang, target_lang)
            except TranslationError:
                return None, 'translation_unavailable'
            if not isinstance(translated, str) or not translated.strip():
                return None, 'translation_unavailable'
            metadata['translation_output_characters'] += len(translated)
            return translated, 'translated'
        finally:
            metadata['translation_seconds'] += time.perf_counter() - started

    @torch.inference_mode()
    def encode(self, texts: list[str], source_lang: str):
        """Retain originals, translate to English, admit conservatively, then encode.

        For English, bypass the backend. Unsupported translations preserve both
        original and translated text locally without a fabricated vector. A plain
        declarative sentence is not rewritten into the finite grammar implicitly.
        """
        source_lang = _language(source_lang)
        if not isinstance(texts, list) or not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError('supply a nonempty list of original source segments')
        sources = list(texts)
        translations, records, supported_indices = [], [], []
        metadata = self._metadata()
        for index, text in enumerate(sources):
            if source_lang == 'en':
                translated, translation_status = text, 'bypassed_english'
            else:
                translated, translation_status = self._translate(text, source_lang, 'en', metadata)
            translations.append(translated)
            if translated is None:
                status = translation_status
            elif is_supported(translated):
                status = 'encoded'
                supported_indices.append(index)
            else:
                status = 'out_of_scope'
            records.append(dict(index=index, status=status, source=text,
                                source_lang=source_lang, translation=translated,
                                translation_lang='en', translation_status=translation_status))
        packet = None
        if supported_indices:
            started = time.perf_counter()
            self.model.eval()
            tokens, lengths = tokenize([translations[index] for index in supported_indices])
            packet = pack_vectors(self.model.encode(tokens, lengths))
            metadata['model_seconds'] += time.perf_counter() - started
        metadata.update(wire_bytes=len(packet) if packet is not None else 0,
                        payload_bytes=64 * len(supported_indices))
        return dict(sources=sources, translations=translations, records=records,
                    packet=packet, supported_indices=supported_indices, metadata=metadata)

    @torch.inference_mode()
    def decode(self, packet: bytes, target_lang: str):
        """Decode bytes only, render the predicted frame, optionally translate.

        There is no original source, translation, gold frame or sender evidence
        argument. Missing target translation returns canonical English explicitly.
        Inconsistent predicted frames are rejected by the renderer rather than
        silently repaired. Sender and receiver must use matching checkpoints.
        """
        target_lang = _language(target_lang)
        metadata = self._metadata()
        self.model.eval()
        started = time.perf_counter()
        frames = receive_packet(self.model, packet)
        metadata['model_seconds'] += time.perf_counter() - started
        canonical_english, texts, records = [], [], []
        for index, frame in enumerate(frames):
            try:
                english = render_meaning(frame)
            except ValueError:
                canonical_english.append(None)
                texts.append(None)
                records.append(dict(index=index, status='invalid_semantic_prediction',
                                    prediction_status='model_prediction', frame=frame,
                                    canonical_english=None, text=None, target_lang=target_lang,
                                    output_lang=None, translation_status='not_attempted'))
                continue
            canonical_english.append(english)
            if target_lang == 'en':
                translated, translation_status = english, 'bypassed_english'
            else:
                translated, translation_status = self._translate(english, 'en', target_lang, metadata)
            available = translated is not None
            text = translated if available else english
            texts.append(text)
            records.append(dict(index=index, status='model_prediction' if available else 'target_translation_unavailable',
                                prediction_status='model_prediction', frame=frame,
                                canonical_english=english, text=text, target_lang=target_lang,
                                output_lang=target_lang if available else 'en',
                                translation_status=translation_status))
        metadata.update(wire_bytes=len(packet), payload_bytes=64 * len(frames),
                        receiver_input='packet_bytes_only')
        return dict(frames=frames, canonical_english=canonical_english,
                    texts=texts, records=records, metadata=metadata)


Gateway = TranslationGateway
