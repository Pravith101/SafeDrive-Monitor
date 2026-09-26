"""Backward-compatible entry point for webcam inference; use live_inference.py."""
from live_inference import run_live_monitor


if __name__ == "__main__":
    run_live_monitor()
