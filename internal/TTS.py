"""
Text-to-Speech (TTS) utilities implemented with Hugging Face Transformers.

Notes
- The LFM2-2.6B model documented in docs/LMF2-2.6B.md is a text model.
  It does not synthesize audio. To provide TTS functionality we rely on
  Transformers' TTS-capable models (e.g., SpeechT5). This module keeps
  a small, reusable API surface ready for later wiring.

Usage
    tts = TTS()  # defaults to SpeechT5
    audio, sr = tts.synthesize("Hello world!")
    tts.save_wav("out.wav", audio, sr)

    # Choose your own HF model id
    tts = TTS(model_id="microsoft/speecht5_tts")
    audio, sr = tts.synthesize("Custom voice", speaker_embedding=emb)
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import importlib.util
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

if "psutil" in sys.modules and getattr(sys.modules["psutil"], "__spec__", None) is None:  # pragma: no cover - test environment fix
    sys.modules["psutil"].__spec__ = importlib.machinery.ModuleSpec("psutil", loader=None)

_transformers_exc: Exception | None = None
if importlib.util.find_spec("torch") is None:
    _transformers_exc = ImportError("PyTorch backend is required for transformers TTS models")

try:
    if _transformers_exc is not None:
        raise _transformers_exc
    from transformers import (
        SpeechT5ForTextToSpeech,
        SpeechT5Processor,
        SpeechT5HifiGan,
        pipeline as hf_pipeline,
    )
except Exception as exc:  # pragma: no cover
    _missing_cause = exc

    class _MissingTransformerDependency:
        _ERROR_MESSAGE = (
            "transformers is required for TTS functionality.\n"
            "Install with: pip install -U transformers"
        )
        _CAUSE = _missing_cause

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(cls._ERROR_MESSAGE) from cls._CAUSE

    SpeechT5ForTextToSpeech = _MissingTransformerDependency  # type: ignore[assignment]
    SpeechT5Processor = _MissingTransformerDependency  # type: ignore[assignment]
    SpeechT5HifiGan = _MissingTransformerDependency  # type: ignore[assignment]

    def hf_pipeline(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(_MissingTransformerDependency._ERROR_MESSAGE) from _missing_cause


@dataclass
class TTSConfig:
    """Configuration for the TTS engine.

    - model_id: TTS-capable HF model id (default: SpeechT5)
    - vocoder_id: Optional HiFi-GAN vocoder for SpeechT5
    - device: 'cpu', 'cuda', or device index like 'cuda:0'
    - torch_dtype: Optional dtype string
    - generate_kwargs: Extra kwargs forwarded to generation
    - sampling_rate: Target sampling rate for output audio
    """

    model_id: str = "microsoft/speecht5_tts"
    vocoder_id: Optional[str] = "microsoft/speecht5_hifigan"
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    generate_kwargs: Dict[str, Any] = None  # type: ignore[assignment]
    sampling_rate: int = 16000

    def __post_init__(self) -> None:
        if self.generate_kwargs is None:
            self.generate_kwargs = {}


class TTS:
    """High-level TTS wrapper using Transformers SpeechT5 pipeline.

    Provides text-to-speech synthesis returning a waveform ndarray and
    sampling rate. Saving helpers are included for convenience.
    """

    def __init__(self, config: Optional[TTSConfig] = None, *, model_id: Optional[str] = None) -> None:
        self.config = config or TTSConfig()
        if model_id:
            self.config.model_id = model_id
        self._pipeline = None
        self._pipeline_error: Exception | None = None

    def _build_pipeline(self):
        load_kwargs: Dict[str, Any] = {}
        if self.config.torch_dtype:
            try:
                import torch  # type: ignore

                dtype_map = {
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }
                load_kwargs["torch_dtype"] = dtype_map.get(str(self.config.torch_dtype).lower())
            except Exception:
                load_kwargs["torch_dtype"] = self.config.torch_dtype

        model = SpeechT5ForTextToSpeech.from_pretrained(self.config.model_id, **{k: v for k, v in load_kwargs.items() if v is not None})
        processor = SpeechT5Processor.from_pretrained(self.config.model_id)

        pipe_kwargs: Dict[str, Any] = {
            "model": model,
            "feature_extractor": processor.feature_extractor if hasattr(processor, "feature_extractor") else processor,
            "tokenizer": processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            "forward_params": {**self.config.generate_kwargs},
        }
        if self.config.device:
            pipe_kwargs["device"] = self.config.device
        if self.config.vocoder_id:
            vocoder = SpeechT5HifiGan.from_pretrained(self.config.vocoder_id)
            pipe_kwargs["vocoder"] = vocoder

        return hf_pipeline("text-to-speech", **pipe_kwargs)

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            self._pipeline = self._build_pipeline()
        except Exception as exc:  # pragma: no cover - surfaced to caller
            self._pipeline_error = exc
            raise
        return self._pipeline

    def synthesize(
        self,
        text: str,
        *,
        speaker_embedding: Optional[np.ndarray] = None,
        language: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech audio from text.

        Parameters
        - text: input text
        - speaker_embedding: optional embedding (1, 512) for SpeechT5 voice
        - language: optional language tag if supported by the model

        Returns
        - (audio: np.ndarray[float32], sampling_rate: int)
        """
        pipe_kwargs: Dict[str, Any] = {}
        if speaker_embedding is not None:
            pipe_kwargs["speaker_embeddings"] = speaker_embedding
        if language:
            pipe_kwargs["language"] = language
        pipeline = self._ensure_pipeline()
        out = pipeline(text, **pipe_kwargs)

        # HF returns dict with 'audio' and 'sampling_rate'
        if isinstance(out, dict) and "audio" in out:
            audio = out["audio"]
            sr = int(out.get("sampling_rate", self.config.sampling_rate))
            # Ensure correct dtype and shape
            audio = np.asarray(audio, dtype=np.float32).flatten()
            return audio, sr

        # Some versions may return array directly
        if isinstance(out, (list, np.ndarray)):
            audio = np.asarray(out, dtype=np.float32).flatten()
            return audio, self.config.sampling_rate

        raise RuntimeError("Unexpected TTS pipeline output format")

    @staticmethod
    def save_wav(path: str, audio: np.ndarray, sampling_rate: int) -> None:
        """Save a mono float32 waveform to a .wav file.

        Prefers soundfile; falls back to scipy if available.
        """
        try:
            import soundfile as sf  # type: ignore

            sf.write(path, audio, sampling_rate)
            return
        except Exception:
            pass

        try:
            from scipy.io import wavfile  # type: ignore

            wavfile.write(path, sampling_rate, (audio * 32767.0).astype(np.int16))
            return
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Saving WAV requires either 'soundfile' or 'scipy'. Install one of:\n"
                "  pip install soundfile\n  pip install scipy"
            ) from exc
