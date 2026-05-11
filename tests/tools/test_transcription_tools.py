"""Tests for tools.transcription_tools — three-provider STT pipeline.

Covers the full provider matrix (local, groq, openai), fallback chains,
model auto-correction, config loading, validation edge cases, and
end-to-end dispatch.  All external dependencies are mocked.
"""

import os
import struct
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(silence)

    return str(wav_path)


@pytest.fixture
def sample_ogg(tmp_path):
    """Create a fake OGG file for validation tests."""
    ogg_path = tmp_path / "test.ogg"
    ogg_path.write_bytes(b"fake audio data")
    return str(ogg_path)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real API keys leak into tests."""
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)


# ============================================================================
# _get_provider — full permutation matrix
# ============================================================================

class TestGetProviderGroq:
    """Groq-specific provider selection tests."""

    def test_groq_when_key_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "groq"

    def test_groq_explicit_no_fallback(self, monkeypatch):
        """Explicit groq with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "none"

    def test_groq_nothing_available(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "none"


class TestGetProviderFallbackPriority:
    """Auto-detect fallback priority and explicit provider behaviour."""

    def test_auto_detect_prefers_local(self):
        """Auto-detect prefers local over any cloud provider."""
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"

    def test_auto_detect_prefers_groq_over_openai(self, monkeypatch):
        """Auto-detect: groq (free) is preferred over openai (paid)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "groq"

    def test_explicit_openai_no_key_returns_none(self, monkeypatch):
        """Explicit openai with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "none"

    def test_unknown_provider_passed_through(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "custom-endpoint"}) == "custom-endpoint"

    def test_empty_config_defaults_to_local(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"


# ============================================================================
# Explicit provider config respected  (GH-1774)
# ============================================================================

class TestExplicitProviderRespected:
    """When stt.provider is explicitly set, that choice is authoritative.
    No silent fallback to a different cloud provider."""

    def test_explicit_local_no_fallback_to_openai(self, monkeypatch):
        """GH-1774: provider=local must not silently fall back to openai
        even when an OpenAI API key is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "***")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "none", f"Expected 'none' but got {result!r}"

    def test_explicit_local_no_fallback_to_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "none"

    def test_explicit_local_uses_local_command_fallback(self, monkeypatch):
        """Local-to-local_command fallback is fine — both are local."""
        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --output_dir {output_dir} --language {language}",
        )
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "local_command"

    def test_explicit_groq_no_fallback_to_openai(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "groq"})
            assert result == "none"

    def test_explicit_openai_no_fallback_to_groq(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "openai"})
            assert result == "none"

    def test_auto_detect_still_falls_back_to_cloud(self, monkeypatch):
        """When no provider is explicitly set, auto-detect cloud fallback works."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            # Empty dict = no explicit provider, uses DEFAULT_PROVIDER auto-detect
            result = _get_provider({})
            assert result == "openai"

    def test_auto_detect_prefers_groq_over_openai(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({})
            assert result == "groq"


# ============================================================================
# _transcribe_groq
# ============================================================================

class TestTranscribeGroq:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_groq
        result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "GROQ_API_KEY" in result["error"]

    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert result["provider"] == "groq"
        mock_client.close.assert_called_once()

    def test_whitespace_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello world  \n"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["transcript"] == "hello world"

    def test_uses_groq_base_url(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client) as mock_openai_cls:
            from tools.transcription_tools import _transcribe_groq, GROQ_BASE_URL
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        call_kwargs = mock_openai_cls.call_args
        assert call_kwargs.kwargs["base_url"] == GROQ_BASE_URL

    def test_api_error_returns_failure(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is False
        assert "API error" in result["error"]
        mock_client.close.assert_called_once()

    def test_permission_error(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = PermissionError("denied")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ============================================================================
# _transcribe_openai — additional tests
# ============================================================================

class TestTranscribeOpenAIExtended:
    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai("/tmp/test.ogg", "whisper-1")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_uses_openai_base_url(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client) as mock_openai_cls:
            from tools.transcription_tools import _transcribe_openai, OPENAI_BASE_URL
            _transcribe_openai(sample_wav, "whisper-1")

        call_kwargs = mock_openai_cls.call_args
        assert call_kwargs.kwargs["base_url"] == OPENAI_BASE_URL

    def test_whitespace_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello  \n"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1")

        assert result["transcript"] == "hello"
        mock_client.close.assert_called_once()

    def test_permission_error(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = PermissionError("denied")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1")

        assert result["success"] is False
        assert "Permission denied" in result["error"]
        mock_client.close.assert_called_once()


class TestTranscribeLocalCommand:
    def test_auto_detects_local_whisper_binary(self, monkeypatch):
        monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
        monkeypatch.setattr("tools.transcription_tools._find_whisper_binary", lambda: "/opt/homebrew/bin/whisper")

        from tools.transcription_tools import _get_local_command_template

        template = _get_local_command_template()

        assert template is not None
        assert template.startswith("/opt/homebrew/bin/whisper ")
        assert "{model}" in template
        assert "{output_dir}" in template

    def test_command_fallback_with_template(self, monkeypatch, sample_ogg, tmp_path):
        out_dir = tmp_path / "local-out"
        out_dir.mkdir()

        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --model {model} --output_dir {output_dir} --language {language}",
        )
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "en")

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self_inner):
                    return str(out_dir)

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _TempDir()

        def fake_run(cmd, *args, **kwargs):
            if isinstance(cmd, list):
                output_path = cmd[-1]
                with open(output_path, "wb") as handle:
                    handle.write(b"RIFF....WAVEfmt ")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            (out_dir / "test.txt").write_text("hello from local command\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("tools.transcription_tools.tempfile.TemporaryDirectory", fake_tempdir)
        monkeypatch.setattr("tools.transcription_tools._find_ffmpeg_binary", lambda: "/opt/homebrew/bin/ffmpeg")
        monkeypatch.setattr("tools.transcription_tools.subprocess.run", fake_run)

        from tools.transcription_tools import _transcribe_local_command

        result = _transcribe_local_command(sample_ogg, "base")

        assert result["success"] is True
        assert result["transcript"] == "hello from local command"
        assert result["provider"] == "local_command"


# ============================================================================
# _transcribe_local — additional tests
# ============================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faster_whisper"),
    reason="faster_whisper not installed",
)
class TestTranscribeLocalExtended:
    def test_model_reuse_on_second_call(self, tmp_path):
        """Second call with same model should NOT reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "base")

        # WhisperModel should be created only once
        assert mock_whisper_cls.call_count == 1

    def test_model_reloaded_on_change(self, tmp_path):
        """Switching model name should reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "small")

        assert mock_whisper_cls.call_count == 2

    def test_exception_returns_failure(self, tmp_path):
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_whisper_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "large-v3")

        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]

    def test_multiple_segments_joined(self, tmp_path):
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg1 = MagicMock()
        seg1.text = "Hello"
        seg2 = MagicMock()
        seg2.text = " world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", return_value=mock_model), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"

    def test_load_time_cuda_lib_failure_falls_back_to_cpu(self, tmp_path):
        """Missing libcublas at load time → reload on CPU, succeed."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "hi"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([seg], info)

        call_args = []

        def fake_whisper(model_name, device, compute_type):
            call_args.append((device, compute_type))
            if device == "auto":
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            return cpu_model

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", side_effect=fake_whisper), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "hi"
        assert call_args == [("auto", "auto"), ("cpu", "int8")]

    def test_runtime_cuda_lib_failure_evicts_cache_and_retries_on_cpu(self, tmp_path):
        """libcublas dlopen fails at transcribe() → evict cache, reload CPU, retry."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "recovered"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        # First model loads fine (auto), but transcribe() blows up on dlopen
        gpu_model = MagicMock()
        gpu_model.transcribe.side_effect = RuntimeError(
            "Library libcublas.so.12 is not found or cannot be loaded"
        )
        # Second model (forced CPU) works
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([seg], info)

        models = [gpu_model, cpu_model]
        call_args = []

        def fake_whisper(model_name, device, compute_type):
            call_args.append((device, compute_type))
            return models.pop(0)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", side_effect=fake_whisper), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "recovered"
        # First load is auto, retry forces CPU.
        assert call_args == [("auto", "auto"), ("cpu", "int8")]
        # Cached-bad-model eviction: the broken GPU model was called once,
        # then discarded; the CPU model served the retry.
        assert gpu_model.transcribe.call_count == 1
        assert cpu_model.transcribe.call_count == 1

    def test_cuda_out_of_memory_does_not_trigger_cpu_fallback(self, tmp_path):
        """'CUDA out of memory' is a real error, not a missing lib — surface it."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_whisper_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        # Single call — no CPU retry, because OOM isn't a missing-lib symptom.
        assert mock_whisper_cls.call_count == 1
        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]


# ============================================================================
# Model auto-correction
# ============================================================================

class TestModelAutoCorrection:
    def test_groq_corrects_openai_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "whisper-1")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_groq_corrects_gpt4o_transcribe(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "gpt-4o-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_openai_corrects_groq_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai, DEFAULT_STT_MODEL
            _transcribe_openai(sample_wav, "whisper-large-v3-turbo")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_openai_corrects_distil_whisper(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai, DEFAULT_STT_MODEL
            _transcribe_openai(sample_wav, "distil-whisper-large-v3-en")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_compatible_groq_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-large-v3"

    def test_compatible_openai_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "gpt-4o-mini-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini-transcribe"

    def test_unknown_model_passes_through_groq(self, monkeypatch, sample_wav):
        """A model not in either known set should not be overridden."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"

    def test_unknown_model_passes_through_openai(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"


# ============================================================================
# _load_stt_config
# ============================================================================

class TestLoadSttConfig:
    def test_returns_dict_when_import_fails(self):
        with patch("tools.transcription_tools._load_stt_config") as mock_load:
            mock_load.return_value = {}
            from tools.transcription_tools import _load_stt_config
            assert _load_stt_config() == {}

    def test_real_load_returns_dict(self):
        """_load_stt_config should always return a dict, even on import error."""
        with patch.dict("sys.modules", {"hermes_cli": None, "hermes_cli.config": None}):
            from tools.transcription_tools import _load_stt_config
            result = _load_stt_config()
        assert isinstance(result, dict)


# ============================================================================
# _validate_audio_file — edge cases
# ============================================================================

class TestValidateAudioFileEdgeCases:
    def test_directory_is_not_a_file(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        # tmp_path itself is a directory with an .ogg-ish name? No.
        # Create a directory with a valid audio extension
        d = tmp_path / "audio.ogg"
        d.mkdir()
        result = _validate_audio_file(str(d))
        assert result is not None
        assert "not a file" in result["error"]

    def test_stat_oserror(self, tmp_path):
        f = tmp_path / "test.ogg"
        f.write_bytes(b"data")
        from tools.transcription_tools import _validate_audio_file

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", side_effect=OSError("disk error")):
            result = _validate_audio_file(str(f))

        assert result is not None
        assert "Failed to access" in result["error"]

    def test_all_supported_formats_accepted(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file, SUPPORTED_FORMATS
        for fmt in SUPPORTED_FORMATS:
            f = tmp_path / f"test{fmt}"
            f.write_bytes(b"data")
            assert _validate_audio_file(str(f)) is None, f"Format {fmt} should be accepted"

    def test_case_insensitive_extension(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        f = tmp_path / "test.MP3"
        f.write_bytes(b"data")
        assert _validate_audio_file(str(f)) is None


# ============================================================================
# transcribe_audio — end-to-end dispatch
# ============================================================================

class TestTranscribeAudioDispatch:
    def test_dispatches_to_groq(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "groq"}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi", "provider": "groq"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "groq"
        mock_groq.assert_called_once()

    def test_dispatches_to_local(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_local.assert_called_once()

    def test_dispatches_to_openai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai",
                   return_value={"success": True, "transcript": "hi", "provider": "openai"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_openai.assert_called_once()

    def test_no_provider_returns_error(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]
        assert "faster-whisper" in result["error"]
        assert "GROQ_API_KEY" in result["error"]

    def test_explicit_openai_no_key_returns_error(self, monkeypatch, sample_ogg):
        """Explicit provider=openai with no key returns an error, not a fallback."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]

    def test_invalid_file_short_circuits(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/audio.wav")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_model_override_passed_to_groq(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="whisper-large-v3")

        _, kwargs = mock_groq.call_args
        assert kwargs.get("model_name") or mock_groq.call_args[0][1] == "whisper-large-v3"

    def test_model_override_passed_to_local(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="large-v3")

        assert mock_local.call_args[0][1] == "large-v3"

    def test_default_model_used_when_none(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio, DEFAULT_GROQ_STT_MODEL
            transcribe_audio(sample_ogg, model=None)

        assert mock_groq.call_args[0][1] == DEFAULT_GROQ_STT_MODEL

    def test_config_local_model_used(self, sample_ogg):
        config = {"local": {"model": "small"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_local.call_args[0][1] == "small"

    def test_config_openai_model_used(self, sample_ogg):
        config = {"openai": {"model": "gpt-4o-transcribe"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai",
                   return_value={"success": True, "transcript": "hi"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_openai.call_args[0][1] == "gpt-4o-transcribe"


# ============================================================================
# _transcribe_mistral
# ============================================================================


@pytest.fixture
def mock_mistral_module():
    """Inject a fake mistralai module into sys.modules for testing."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_mistral_cls = MagicMock(return_value=mock_client)
    fake_module = MagicMock()
    fake_module.Mistral = mock_mistral_cls
    with patch.dict("sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}):
        yield mock_client


class TestTranscribeMistral:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral("/tmp/test.ogg", "voxtral-mini-latest")
        assert result["success"] is False
        assert "MISTRAL_API_KEY" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        mock_result = MagicMock()
        mock_result.text = "hello from mistral"
        mock_mistral_module.audio.transcriptions.complete.return_value = mock_result

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is True
        assert result["transcript"] == "hello from mistral"
        assert result["provider"] == "mistral"
        mock_mistral_module.audio.transcriptions.complete.assert_called_once()
        mock_mistral_module.__exit__.assert_called_once()

    def test_api_error_returns_failure(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.transcriptions.complete.side_effect = RuntimeError("secret-key-leaked")

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is False
        assert "RuntimeError" in result["error"]
        assert "secret-key-leaked" not in result["error"]

    def test_permission_error(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.transcriptions.complete.side_effect = PermissionError("denied")

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ============================================================================
# _get_provider — Mistral
# ============================================================================

class TestGetProviderMistral:
    """Mistral-specific provider selection tests."""

    def test_mistral_when_key_and_sdk_available(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "mistral"

    def test_mistral_explicit_no_key_returns_none(self, monkeypatch):
        """Explicit mistral with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "none"

    def test_mistral_explicit_no_sdk_returns_none(self, monkeypatch):
        """Explicit mistral with key but no SDK returns none."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "none"

    def test_auto_detect_mistral_after_openai(self, monkeypatch):
        """Auto-detect: mistral is tried after openai when both are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "mistral"

    def test_auto_detect_openai_preferred_over_mistral(self, monkeypatch):
        """Auto-detect: openai is preferred over mistral (both paid, openai more common)."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "openai"

    def test_auto_detect_groq_preferred_over_mistral(self, monkeypatch):
        """Auto-detect: groq (free) is preferred over mistral (paid)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "groq"

    def test_auto_detect_skips_mistral_without_sdk(self, monkeypatch):
        """Auto-detect: mistral skipped when key is set but SDK is not installed."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "none"


# ============================================================================
# transcribe_audio — Mistral dispatch
# ============================================================================

class TestTranscribeAudioMistralDispatch:
    def test_dispatches_to_mistral(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "mistral"}), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi", "provider": "mistral"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "mistral"
        mock_mistral.assert_called_once()

    def test_config_mistral_model_used(self, sample_ogg):
        config = {"provider": "mistral", "mistral": {"model": "voxtral-mini-2602"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_mistral.call_args[0][1] == "voxtral-mini-2602"

    def test_model_override_passed_to_mistral(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="voxtral-mini-2602")

        assert mock_mistral.call_args[0][1] == "voxtral-mini-2602"


# ============================================================================
# _transcribe_xai
# ============================================================================


@pytest.fixture
def mock_xai_http_module():
    """Inject a fake tools.xai_http module for testing."""
    fake_module = MagicMock()
    fake_module.hermes_xai_user_agent = MagicMock(return_value="hermes-xai/test")
    with patch.dict("sys.modules", {"tools.xai_http": fake_module}):
        yield fake_module


class TestTranscribeXAI:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_xai
        result = _transcribe_xai("/tmp/test.ogg", "grok-stt")
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "bonjour le monde",
            "language": "fr",
            "duration": 3.2,
        }

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is True
        assert result["transcript"] == "bonjour le monde"
        assert result["provider"] == "xai"

    def test_whitespace_stripped(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "  hello world  \n"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["transcript"] == "hello world"

    def test_api_error_returns_failure(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid audio format"}}
        mock_response.text = '{"error": {"message": "Invalid audio format"}}'

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "HTTP 400" in result["error"]
        assert "Invalid audio format" in result["error"]

    def test_empty_transcript_returns_failure(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "   "}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "empty transcript" in result["error"]

    def test_permission_error(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("builtins.open", side_effect=PermissionError("denied")):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    def test_network_error_returns_failure(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", side_effect=ConnectionError("timeout")):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "timeout" in result["error"]

    def test_sends_language_and_format(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        # Explicitly set language via env to exercise the override chain
        # (config > env > DEFAULT_LOCAL_STT_LANGUAGE)
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "fr")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "fr", "duration": 1.0}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
        assert data.get("language") == "fr"
        assert data.get("format") == "true"

    def test_custom_base_url(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        monkeypatch.setenv("XAI_STT_BASE_URL", "https://custom.x.ai/v1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "en", "duration": 1.0}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "custom.x.ai" in url

    def test_diarize_sent_when_configured(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "fr", "duration": 1.0}

        config = {"xai": {"diarize": True}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        data = mock_post.call_args.kwargs.get("data", mock_post.call_args[1].get("data", {}))
        assert data.get("diarize") == "true"


# ============================================================================
# _get_provider — xAI
# ============================================================================

class TestGetProviderXAI:
    """xAI-specific provider selection tests."""

    def test_xai_when_key_set(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "xai"}) == "xai"

    def test_xai_explicit_no_key_returns_none(self, monkeypatch):
        """Explicit xai with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "xai"}) == "none"

    def test_auto_detect_xai_after_mistral(self, monkeypatch):
        """Auto-detect: xai is tried after mistral when all above are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "xai"

    def test_auto_detect_mistral_preferred_over_xai(self, monkeypatch):
        """Auto-detect: mistral is preferred over xai."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "mistral"

    def test_auto_detect_no_key_returns_none(self, monkeypatch):
        """Auto-detect: xai skipped when no key is set."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "none"


# ============================================================================
# transcribe_audio — xAI dispatch
# ============================================================================

class TestTranscribeAudioXAIDispatch:
    def test_dispatches_to_xai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "xai"}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi", "provider": "xai"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "xai"
        mock_xai.assert_called_once()

    def test_model_default_is_grok_stt(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "xai"}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_xai.call_args[0][1] == "grok-stt"

    def test_model_override_passed_to_xai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="custom-stt")

        assert mock_xai.call_args[0][1] == "custom-stt"


# ============================================================================
# Fixtures — Yandex / hybrid_ru
# ============================================================================

@pytest.fixture
def sample_oga(tmp_path):
    """Fake OGA (Telegram voice) file — 512 bytes, well under 1 MB."""
    oga_path = tmp_path / "voice.oga"
    oga_path.write_bytes(b"\x4f\x67\x67\x53" + b"\x00" * 508)
    return str(oga_path)


# ============================================================================
# _get_audio_duration_seconds / _get_audio_channels helpers
# ============================================================================

class TestGetAudioChannels:
    def test_get_audio_channels_via_ffprobe_mock(self, sample_ogg):
        mock_result = MagicMock()
        mock_result.stdout = "2\n"

        with patch("tools.transcription_tools._find_binary", return_value="/usr/bin/ffprobe"), \
             patch("tools.transcription_tools.subprocess.run", return_value=mock_result):
            from tools.transcription_tools import _get_audio_channels
            result = _get_audio_channels(sample_ogg)

        assert result == 2

    def test_returns_none_when_ffprobe_missing(self, sample_ogg):
        with patch("tools.transcription_tools._find_binary", return_value=None):
            from tools.transcription_tools import _get_audio_channels
            result = _get_audio_channels(sample_ogg)

        assert result is None

    def test_returns_none_on_empty_output(self, sample_ogg):
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("tools.transcription_tools._find_binary", return_value="/usr/bin/ffprobe"), \
             patch("tools.transcription_tools.subprocess.run", return_value=mock_result):
            from tools.transcription_tools import _get_audio_channels
            result = _get_audio_channels(sample_ogg)

        assert result is None


# ============================================================================
# _ensure_mono_for_yandex
# ============================================================================

class TestEnsureMonoForYandex:
    def test_ensure_mono_passes_through_mono_file(self, sample_ogg, tmp_path):
        with patch("tools.transcription_tools._get_audio_channels", return_value=1):
            from tools.transcription_tools import _ensure_mono_for_yandex
            path, error = _ensure_mono_for_yandex(sample_ogg, str(tmp_path))

        assert path == sample_ogg
        assert error is None

    def test_ensure_mono_passes_through_when_channels_unknown(self, sample_ogg, tmp_path):
        with patch("tools.transcription_tools._get_audio_channels", return_value=None):
            from tools.transcription_tools import _ensure_mono_for_yandex
            path, error = _ensure_mono_for_yandex(sample_ogg, str(tmp_path))

        assert path == sample_ogg
        assert error is None

    def test_ensure_mono_downmixes_stereo(self, sample_ogg, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("tools.transcription_tools._get_audio_channels", return_value=2), \
             patch("tools.transcription_tools._find_binary", return_value="/usr/bin/ffmpeg"), \
             patch("tools.transcription_tools.subprocess.run", return_value=mock_result):
            from tools.transcription_tools import _ensure_mono_for_yandex
            path, error = _ensure_mono_for_yandex(sample_ogg, str(tmp_path))

        assert "yandex-mono" in path
        assert path.endswith(".ogg")
        assert error is None

    def test_ensure_mono_no_ffmpeg_returns_error_for_stereo(self, sample_ogg, tmp_path):
        with patch("tools.transcription_tools._get_audio_channels", return_value=2), \
             patch("tools.transcription_tools._find_binary", return_value=None):
            from tools.transcription_tools import _ensure_mono_for_yandex
            path, error = _ensure_mono_for_yandex(sample_ogg, str(tmp_path))

        assert path == sample_ogg
        assert error is not None
        assert "ffmpeg" in error.lower() or "ffmpeg" in error


# ============================================================================
# _transcribe_yandex
# ============================================================================

class TestTranscribeYandex:
    def test_no_key(self, monkeypatch, sample_oga):
        monkeypatch.delenv("YANDEX_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_yandex
        result = _transcribe_yandex(sample_oga, "speechkit-v3")
        assert result["success"] is False
        assert "YANDEX_API_KEY" in result["error"]

    def test_successful_transcription_ogg(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "привет мир"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is True
        assert result["transcript"] == "привет мир"
        assert result["provider"] == "yandex"

    def test_whitespace_stripped(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "  привет  \n"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["transcript"] == "привет"

    def test_file_too_large(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        large_file = tmp_path / "big.ogg"
        large_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post") as mock_post:
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(str(large_file), "speechkit-v3")

        assert result["success"] is False
        assert "МБ" in result["error"] or "large" in result["error"].lower() or "большой" in result["error"]
        mock_post.assert_not_called()

    def test_file_too_long(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=35.0), \
             patch("requests.post") as mock_post:
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is False
        assert "35.0" in result["error"] or "длинное" in result["error"]
        mock_post.assert_not_called()

    def test_duration_check_skipped_when_ffprobe_unavailable(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "тест"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=None), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is True

    def test_http_401_returns_error(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "bad-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error_message": "API key not found"}
        mock_response.text = '{"error_message": "API key not found"}'

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is False
        assert "HTTP 401" in result["error"]
        assert "API key not found" in result["error"]

    def test_http_error_without_json_body(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("no json")
        mock_response.text = "Internal Server Error"

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    def test_empty_transcript_returns_failure(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "  "}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is False
        assert "пустую" in result["error"] or "empty" in result["error"].lower()

    def test_permission_error(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("builtins.open", side_effect=PermissionError("denied")):
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_oga, "speechkit-v3")

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    def test_unsupported_format_wav_is_accepted(self, monkeypatch, sample_wav):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "тест"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=1.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_wav, None)), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(sample_wav, "speechkit-v3")

        assert result["success"] is True
        call_kwargs = mock_post.call_args
        params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
        assert params.get("format") == "lpcm"

    def test_unsupported_format_mp3_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        mp3_file = tmp_path / "audio.mp3"
        mp3_file.write_bytes(b"fake mp3 data")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("requests.post") as mock_post:
            from tools.transcription_tools import _transcribe_yandex
            result = _transcribe_yandex(str(mp3_file), "speechkit-v3")

        assert result["success"] is False
        assert "mp3" in result["error"].lower() or "формат" in result["error"]
        mock_post.assert_not_called()

    def test_config_options_propagate_to_params(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        config = {
            "yandex": {
                "folder_id": "b1g12345",
                "profanity_filter": True,
                "language": "ru-RU",
                "topic": "general",
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "тест"}

        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_yandex
            _transcribe_yandex(sample_oga, "speechkit-v3")

        call_kwargs = mock_post.call_args
        params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
        assert params.get("folderId") == "b1g12345"
        assert params.get("profanityFilter") == "true"

    def test_auth_header_is_api_key_not_bearer(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "my-secret-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "тест"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._ensure_mono_for_yandex",
                   return_value=(sample_oga, None)), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_yandex
            _transcribe_yandex(sample_oga, "speechkit-v3")

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert headers["Authorization"].startswith("Api-Key ")
        assert not headers["Authorization"].startswith("Bearer ")


# ============================================================================
# _get_provider — Yandex
# ============================================================================

class TestGetProviderYandex:
    def test_yandex_when_key_set(self, monkeypatch):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "yandex"}) == "yandex"

    def test_yandex_explicit_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("YANDEX_API_KEY", raising=False)
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "yandex"}) == "none"

    def test_auto_detect_yandex_after_xai(self, monkeypatch):
        """Auto-detect: yandex is tried after xAI when all above are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "yandex"

    def test_auto_detect_xai_preferred_over_yandex(self, monkeypatch):
        """Auto-detect: xAI key takes priority over Yandex in auto-detect (no local)."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-key")
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "xai"

    def test_auto_detect_hybrid_ru_when_local_and_yandex(self, monkeypatch):
        """Auto-detect: hybrid_ru when both faster-whisper and YANDEX_API_KEY present."""
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "hybrid_ru"

    def test_auto_detect_local_when_faster_whisper_no_yandex(self, monkeypatch):
        """Auto-detect: plain local when faster-whisper available but no Yandex key."""
        monkeypatch.delenv("YANDEX_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"


# ============================================================================
# _get_provider — hybrid_ru explicit
# ============================================================================

class TestGetProviderHybridRu:
    def test_hybrid_ru_explicit_with_faster_whisper(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "hybrid_ru"}) == "hybrid_ru"

    def test_hybrid_ru_explicit_no_faster_whisper_returns_none(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "hybrid_ru"}) == "none"


# ============================================================================
# transcribe_audio — Yandex dispatch
# ============================================================================

class TestTranscribeAudioYandexDispatch:
    def test_dispatches_to_yandex(self, sample_oga):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "yandex"}), \
             patch("tools.transcription_tools._get_provider", return_value="yandex"), \
             patch("tools.transcription_tools._transcribe_yandex",
                   return_value={"success": True, "transcript": "привет", "provider": "yandex"}) as mock_yandex:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_oga)

        assert result["success"] is True
        assert result["provider"] == "yandex"
        mock_yandex.assert_called_once()

    def test_config_yandex_model_placeholder_used(self, sample_oga):
        config = {"provider": "yandex", "yandex": {}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="yandex"), \
             patch("tools.transcription_tools._transcribe_yandex",
                   return_value={"success": True, "transcript": "привет"}) as mock_yandex:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_oga, model=None)

        assert mock_yandex.call_args[0][1] == "speechkit-v3"


# ============================================================================
# _transcribe_hybrid_ru
# ============================================================================

class TestTranscribeHybridRu:
    def test_routes_to_yandex_for_short_mono(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        yandex_ok = {"success": True, "transcript": "привет", "provider": "yandex"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=20.0), \
             patch("tools.transcription_tools._get_audio_channels", return_value=1), \
             patch("tools.transcription_tools._transcribe_yandex", return_value=yandex_ok) as mock_yandex, \
             patch("tools.transcription_tools._transcribe_local") as mock_local:
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(sample_oga, "base")

        mock_yandex.assert_called_once()
        mock_local.assert_not_called()
        assert result["routed_to"] == "yandex"
        assert result["success"] is True

    def test_routes_to_local_when_file_too_large(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        large_file = tmp_path / "large.ogg"
        large_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

        local_ok = {"success": True, "transcript": "привет", "provider": "local"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=10.0), \
             patch("tools.transcription_tools._get_audio_channels", return_value=1), \
             patch("tools.transcription_tools._transcribe_yandex") as mock_yandex, \
             patch("tools.transcription_tools._transcribe_local", return_value=local_ok) as mock_local:
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(str(large_file), "base")

        mock_yandex.assert_not_called()
        mock_local.assert_called_once()
        assert result["routed_to"] == "local"

    def test_routes_to_local_when_duration_over_28s(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        local_ok = {"success": True, "transcript": "длинный текст", "provider": "local"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=45.0), \
             patch("tools.transcription_tools._get_audio_channels", return_value=1), \
             patch("tools.transcription_tools._transcribe_yandex") as mock_yandex, \
             patch("tools.transcription_tools._transcribe_local", return_value=local_ok) as mock_local:
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(sample_oga, "base")

        mock_yandex.assert_not_called()
        mock_local.assert_called_once()
        assert result["routed_to"] == "local"

    def test_routes_to_local_when_no_yandex_key(self, monkeypatch, sample_oga):
        monkeypatch.delenv("YANDEX_API_KEY", raising=False)

        local_ok = {"success": True, "transcript": "привет", "provider": "local"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=10.0), \
             patch("tools.transcription_tools._get_audio_channels", return_value=1), \
             patch("tools.transcription_tools._transcribe_yandex") as mock_yandex, \
             patch("tools.transcription_tools._transcribe_local", return_value=local_ok):
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(sample_oga, "base")

        mock_yandex.assert_not_called()
        assert result["routed_to"] == "local"

    def test_yandex_failure_falls_back_to_local(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        yandex_fail = {"success": False, "transcript": "", "error": "API error 500"}
        local_ok = {"success": True, "transcript": "привет", "provider": "local"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=10.0), \
             patch("tools.transcription_tools._get_audio_channels", return_value=1), \
             patch("tools.transcription_tools._transcribe_yandex", return_value=yandex_fail), \
             patch("tools.transcription_tools._transcribe_local", return_value=local_ok) as mock_local:
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(sample_oga, "base")

        mock_local.assert_called_once()
        assert result["routed_to"] == "local"
        assert result["yandex_attempted"] is True
        assert result["yandex_error"] == "API error 500"
        assert result["success"] is True

    def test_yandex_success_no_local_call(self, monkeypatch, sample_oga):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        yandex_ok = {"success": True, "transcript": "привет", "provider": "yandex"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=5.0), \
             patch("tools.transcription_tools._get_audio_channels", return_value=1), \
             patch("tools.transcription_tools._transcribe_yandex", return_value=yandex_ok), \
             patch("tools.transcription_tools._transcribe_local") as mock_local:
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(sample_oga, "base")

        mock_local.assert_not_called()
        assert result["routed_to"] == "yandex"

    def test_ffprobe_unavailable_routes_to_yandex_if_under_size(self, monkeypatch, sample_oga):
        """If duration check returns None (no ffprobe), small file should still try Yandex."""
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")

        yandex_ok = {"success": True, "transcript": "тест", "provider": "yandex"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_audio_duration_seconds", return_value=None), \
             patch("tools.transcription_tools._get_audio_channels", return_value=None), \
             patch("tools.transcription_tools._transcribe_yandex", return_value=yandex_ok) as mock_yandex, \
             patch("tools.transcription_tools._transcribe_local") as mock_local:
            from tools.transcription_tools import _transcribe_hybrid_ru
            result = _transcribe_hybrid_ru(sample_oga, "base")

        mock_yandex.assert_called_once()
        mock_local.assert_not_called()
        assert result["routed_to"] == "yandex"


# ============================================================================
# transcribe_audio — hybrid_ru dispatch
# ============================================================================

class TestTranscribeAudioHybridRuDispatch:
    def test_dispatches_to_hybrid_ru(self, sample_oga):
        hybrid_ok = {"success": True, "transcript": "привет", "provider": "yandex", "routed_to": "yandex"}
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="hybrid_ru"), \
             patch("tools.transcription_tools._transcribe_hybrid_ru",
                   return_value=hybrid_ok) as mock_hybrid:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_oga)

        mock_hybrid.assert_called_once()
        assert result["routed_to"] == "yandex"


# ============================================================================
# Correction F — Local model cache with HF-Hub path
# ============================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faster_whisper"),
    reason="faster_whisper not installed",
)
def test_local_model_cache_with_hf_hub_path(monkeypatch):
    """Loading the same HF-Hub-style model name twice reuses the cached instance."""
    hf_model = "bzikst/faster-whisper-large-v3-russian"

    mock_segment = MagicMock()
    mock_segment.text = "привет"
    mock_info = MagicMock()
    mock_info.language = "ru"
    mock_info.duration = 1.0

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    mock_whisper_cls = MagicMock(return_value=mock_model)

    import tempfile
    import struct
    import wave as _wave

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    n_frames = 8000
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))
    with _wave.open(tmp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(silence)

    try:
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(tmp_wav, hf_model)
            _transcribe_local(tmp_wav, hf_model)

        assert mock_whisper_cls.call_count == 1
        assert mock_whisper_cls.call_args[0][0] == hf_model
    finally:
        import os
        os.unlink(tmp_wav)


# ============================================================================
# Integration test (skipped without real key)
# ============================================================================

@pytest.mark.skipif(
    not os.getenv("YANDEX_API_KEY"),
    reason="YANDEX_API_KEY not set — integration test skipped",
)
def test_yandex_integration_real_ogg(tmp_path):
    """
    Send a real OGG file with Russian speech to Yandex SpeechKit.
    Place /tmp/test_ru_voice.ogg before running:
      YANDEX_API_KEY=... pytest -k test_yandex_integration
    """
    import shutil
    ogg_src = Path("/tmp/test_ru_voice.ogg")  # noqa: F811 — module-level Path imported above
    if not ogg_src.exists():
        pytest.skip("test audio file /tmp/test_ru_voice.ogg not found")
    ogg = tmp_path / "voice.ogg"
    shutil.copy(ogg_src, ogg)
    from tools.transcription_tools import _transcribe_yandex
    result = _transcribe_yandex(str(ogg), "speechkit-v3")
    assert result["success"] is True, result.get("error")
    assert result["transcript"]
    assert any("Ѐ" <= c <= "ӿ" for c in result["transcript"]), "expected Cyrillic"
