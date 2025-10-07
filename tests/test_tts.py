import os
import sys
import types
from unittest.mock import patch

import numpy as np


class FakeTTSPipeline:
    def __call__(self, text, **kwargs):
        assert isinstance(text, str)
        # minimal fake output: short sine-like signal
        audio = np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float32)
        return {"audio": audio, "sampling_rate": 16000}


@patch("internal.TTS.hf_pipeline", return_value=FakeTTSPipeline())
@patch("internal.TTS.SpeechT5HifiGan.from_pretrained", return_value=object())
@patch("internal.TTS.SpeechT5Processor.from_pretrained", return_value=types.SimpleNamespace(tokenizer=object(), feature_extractor=object()))
@patch("internal.TTS.SpeechT5ForTextToSpeech.from_pretrained", return_value=object())
def test_tts_synthesize_returns_audio(_m_model, _m_proc, _m_vocoder, _m_pipe):
    from internal.TTS import TTS

    tts = TTS()
    audio, sr = tts.synthesize("Hello")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert isinstance(sr, int) and sr > 0


def test_save_wav_uses_soundfile(tmp_path):
    # Inject a fake soundfile module
    fake_sf = types.SimpleNamespace()

    def fake_write(path, data, samplerate):
        # create the file to emulate a save
        with open(path, "wb") as f:
            f.write(b"RIFF")

    fake_sf.write = fake_write
    sys.modules["soundfile"] = fake_sf

    from internal.TTS import TTS

    tts = TTS()
    audio = np.zeros(16000, dtype=np.float32)
    out_path = os.path.join(tmp_path, "out.wav")
    tts.save_wav(out_path, audio, 16000)
    assert os.path.exists(out_path)

