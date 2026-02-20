import atexit
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class _TeeStream:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def enable_local_training_file_logging(script_path: str, logs_dir: str = "logs") -> Path | None:
    """Mirror local interactive training output to logs/<script>_<timestamp>.log."""
    stdout_is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    stderr_is_tty = getattr(sys.stderr, "isatty", lambda: False)()
    if not stdout_is_tty and not stderr_is_tty:
        return None

    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_stem = Path(script_path).stem
    log_path = log_dir / f"{script_stem}_{timestamp}.log"
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)

    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = _TeeStream(stdout, log_stream)
    sys.stderr = _TeeStream(stderr, log_stream)

    def _close_log_stream() -> None:
        try:
            log_stream.flush()
        finally:
            log_stream.close()

    atexit.register(_close_log_stream)
    print(f"Saving training logs to {log_path}")
    return log_path
