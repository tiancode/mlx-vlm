#!/usr/bin/env python3
"""Run a service with bounded rotating logs and forward shutdown signals."""

import argparse
import os
import pathlib
import signal
import subprocess
import threading


class RotatingLog:
    def __init__(self, path, max_bytes=64 * 1024 * 1024, backups=3):
        if max_bytes < 1 or backups < 1:
            raise ValueError("log size and backup count must be positive")
        self.path = pathlib.Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._open()

    def _open(self):
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._file = os.fdopen(fd, "ab", buffering=0)
        self.size = os.fstat(fd).st_size

    def _rotate(self):
        self._file.close()
        for index in range(self.backups, 1, -1):
            source = pathlib.Path(f"{self.path}.{index - 1}")
            if source.exists():
                os.replace(source, f"{self.path}.{index}")
        os.replace(self.path, f"{self.path}.1")
        self._open()

    def write(self, data):
        # Bound even a single huge line; ordinary lines stay together.
        for offset in range(0, len(data), self.max_bytes):
            chunk = data[offset:offset + self.max_bytes]
            if self.size and self.size + len(chunk) > self.max_bytes:
                self._rotate()
            self._file.write(chunk)
            self.size += len(chunk)

    def close(self):
        if self._file is not None:
            self._file.close()


def run(command, log):
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    timer = None

    def signal_group(signum):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    def stop(signum, _frame):
        nonlocal timer
        signal_group(signum)
        if timer is None:
            timer = threading.Timer(30, signal_group, args=(signal.SIGKILL,))
            timer.daemon = True
            timer.start()

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while chunk := child.stdout.readline(256 * 1024):
            log.write(chunk)
        status = child.wait()
        return status if status >= 0 else 128 - status
    finally:
        if child.poll() is None:
            signal_group(signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_group(signal.SIGKILL)
                child.wait()
        child.stdout.close()
        if timer is not None:
            timer.cancel()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--max-bytes", type=int,
                        default=os.environ.get("SERVICE_LOG_MAX_BYTES", 64 * 1024 * 1024))
    parser.add_argument("--backups", type=int,
                        default=os.environ.get("SERVICE_LOG_BACKUPS", 3))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_bytes < 1 or args.backups < 1:
        parser.error("a command and positive log limits are required")
    log = RotatingLog(args.log, args.max_bytes, args.backups)
    try:
        return run(command, log)
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
