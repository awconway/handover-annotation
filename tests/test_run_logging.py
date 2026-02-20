import io
import sys

from training.run_logging import enable_local_training_file_logging


class _FakeStream(io.StringIO):
    def __init__(self, is_tty: bool):
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_enable_local_training_file_logging_writes_stdout_and_stderr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdout", _FakeStream(is_tty=True))
    monkeypatch.setattr(sys, "stderr", _FakeStream(is_tty=True))

    log_path = enable_local_training_file_logging("run_train_checklist.py")

    assert log_path is not None
    print("hello from stdout")
    sys.stderr.write("hello from stderr\n")
    sys.stdout.flush()
    sys.stderr.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "Saving training logs to" in content
    assert "hello from stdout" in content
    assert "hello from stderr" in content


def test_enable_local_training_file_logging_skips_non_interactive_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdout", _FakeStream(is_tty=False))
    monkeypatch.setattr(sys, "stderr", _FakeStream(is_tty=False))

    log_path = enable_local_training_file_logging("run_train_checklist.py")

    assert log_path is None
    assert not (tmp_path / "logs").exists()
