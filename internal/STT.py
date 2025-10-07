"""
Speech-to-Text (STT) utilities implemented with Hugging Face Transformers.

Notes
- The LFM2-2.6B model documented in docs/LMF2-2.6B.md exposes a
  text-generation interface via `transformers`. It does not natively
  accept audio inputs. To provide full STT functionality, this module
  uses Transformers' ASR models (e.g., Whisper or Wav2Vec2) under the
  hood while keeping a small, clean API surface for later wiring.

- No external wiring or app integration is performed here; this is a
  focused, reusable STT component that can be instantiated and called
  directly by the rest of the system when needed.

Usage
    stt = STT()  # defaults to a reasonable Whisper model
    text = stt.transcribe("path/to/audio.wav")

    # Or pick your own HF model id (must be ASR-capable)
    stt = STT(model_id="openai/whisper-small")
    text = stt.transcribe(audio_array, sampling_rate=16000)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline as hf_pipeline,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for STT functionality.\n"
        "Install with: pip install -U transformers"
    ) from exc


AudioInput = Union[str, np.ndarray]


@dataclass
class STTConfig:
    """Configuration for the STT engine.

    - model_id: Hugging Face model id supporting ASR (e.g., Whisper/Wav2Vec2).
    - device: 'cpu', 'cuda', or device index like 'cuda:0'.
    - torch_dtype: Optional dtype string passed to model load (e.g., 'float16', 'bfloat16').
    - chunk_length_s: Optional chunk size for long audio in seconds (Whisper supports this).
    - batch_size: Pipeline batch size for batched inputs.
    - generate_kwargs: Extra kwargs forwarded to the underlying generate/decoding.
    """

    model_id: str = "openai/whisper-small"
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    chunk_length_s: Optional[float] = None
    batch_size: int = 1
    generate_kwargs: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.generate_kwargs is None:
            self.generate_kwargs = {}


class STT:
    """High-level STT wrapper around a Transformers ASR pipeline.

    This intentionally does not wire into any external runtime. It simply
    offers programmatic transcription for an audio file path or a raw 
    waveform array.
    """

    def __init__(self, config: Optional[STTConfig] = None, *, model_id: Optional[str] = None) -> None:
        self.config = config or STTConfig()
        if model_id:
            self.config.model_id = model_id

        self._pipeline = self._build_pipeline()

    def _build_pipeline(self):
        load_kwargs: Dict[str, Any] = {}
        if self.config.device:
            load_kwargs["device_map"] = "auto" if self.config.device == "auto" else None
        if self.config.torch_dtype:
            # Try to convert common dtype strings to torch dtypes
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
                # Fallback: let transformers attempt to interpret
                load_kwargs["torch_dtype"] = self.config.torch_dtype

        # Load model + processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.config.model_id, **{k: v for k, v in load_kwargs.items() if v is not None})
        processor = AutoProcessor.from_pretrained(self.config.model_id)

        # Build pipeline
        pipe_kwargs: Dict[str, Any] = {
            "model": model,
            "tokenizer": processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            "feature_extractor": processor.feature_extractor if hasattr(processor, "feature_extractor") else processor,
            "batch_size": self.config.batch_size,
            "generate_kwargs": self.config.generate_kwargs,
        }
        if self.config.chunk_length_s is not None:
            pipe_kwargs["chunk_length_s"] = self.config.chunk_length_s

        # Device selection: let the pipeline auto-place if device not specified explicitly
        if self.config.device:
            pipe_kwargs["device"] = self.config.device

        return hf_pipeline("automatic-speech-recognition", **pipe_kwargs)

    def transcribe(
        self,
        audio: AudioInput,
        *,
        sampling_rate: Optional[int] = None,
        return_timestamps: bool = False,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe an audio file path or waveform into text.

        Parameters
        - audio: str path to an audio file, or a numpy ndarray waveform (float32) of shape (T,) or (T, C)
        - sampling_rate: required when passing a numpy array
        - return_timestamps: include timestamp segments if the underlying model supports it
        - language: optional language code hint (e.g., 'en') for multilingual ASR models

        Returns
        - dict with at least {'text': str}. May include 'chunks' when timestamps requested
        """

        pipe_inputs: Dict[str, Any] = {}
        if isinstance(audio, np.ndarray):
            if sampling_rate is None:
                raise ValueError("sampling_rate must be provided when passing a numpy array")
            pipe_inputs["sampling_rate"] = sampling_rate
        if return_timestamps:
            pipe_inputs["return_timestamps"] = True
        if language:
            pipe_inputs["language"] = language

        result = self._pipeline(audio, **pipe_inputs)
        # Ensure a consistent dict output
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result:
            # Some pipelines may return a list when batching
            return result[0]
        return {"text": ""}
