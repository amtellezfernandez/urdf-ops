"""Utilities for LeRobot Dataset Wrapper."""

import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_output():
    """Suppress stdout and stderr at OS level."""
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)

        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        if 'devnull_fd' in locals():
            os.close(devnull_fd)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
