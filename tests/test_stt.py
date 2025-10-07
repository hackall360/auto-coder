import types
from unittest.mock import patch

import numpy as np


def make_fake_processor():
    # Minimal processor with tokenizer and feature_extractor attributes
    proc = types.SimpleNamespace()
    proc.tokenizer = object()
    proc.feature_extractor = object()
    return proc


class FakeASRPipeline:
    def __call__(self, audio, **kwargs):
        # Simulate different outputs based on kwargs
        text = "hello world"
        if kwargs.get("return_timestamps"):
            return {"text": text, "chunks": [{"text": text, "timestamp": (0.0, 1.0)}]}
        return {"text": text}


@patch("internal.STT.hf_pipeline", return_value=FakeASRPipeline())
@patch("internal.STT.AutoProcessor.from_pretrained", return_value=make_fake_processor())
@patch("internal.STT.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=object())
def test_stt_transcribe_file(_m_model, _m_proc, _m_pipe):
    from internal.STT import STT

    stt = STT()
    out = stt.transcribe("dummy.wav", return_timestamps=True, language="en")
    assert isinstance(out, dict)
    assert "text" in out and isinstance(out["text"], str)
    assert "chunks" in out and isinstance(out["chunks"], list)


@patch("internal.STT.hf_pipeline", return_value=FakeASRPipeline())
@patch("internal.STT.AutoProcessor.from_pretrained", return_value=make_fake_processor())
@patch("internal.STT.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=object())
def test_stt_transcribe_array_requires_sr(_m_model, _m_proc, _m_pipe):
    from internal.STT import STT

    stt = STT()
    audio = np.zeros(16000, dtype=np.float32)
    try:
        stt.transcribe(audio)  # missing sampling_rate
        assert False, "Expected ValueError for missing sampling_rate"
    except ValueError:
        pass


@patch("internal.STT.hf_pipeline", return_value=FakeASRPipeline())
@patch("internal.STT.AutoProcessor.from_pretrained", return_value=make_fake_processor())
@patch("internal.STT.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=object())
def test_stt_transcribe_array_ok(_m_model, _m_proc, _m_pipe):
    from internal.STT import STT

    stt = STT()
    audio = np.zeros(16000, dtype=np.float32)
    out = stt.transcribe(audio, sampling_rate=16000)
    assert isinstance(out, dict)
    assert out.get("text") == "hello world"

