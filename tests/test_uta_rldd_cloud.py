import numpy as np

from models.uta_rldd_cloud import choose_threshold, discover_videos, summarize, video_rows


def test_discover_videos_keeps_official_participant_folds(tmp_path):
    for fold in range(1, 6):
        for person in range((fold - 1) * 12 + 1, fold * 12 + 1):
            folder = tmp_path / f"Fold{fold}_part1" / f"Fold{fold}_part1" / f"{person:02d}"
            folder.mkdir(parents=True)
            for label in ("0", "5", "10"):
                (folder / f"{label}.mp4").touch()

    videos = discover_videos(tmp_path)

    assert len(videos) == 180
    assert len({row["participant"] for row in videos}) == 60
    assert {row["fold"] for row in videos} == {1, 2, 3, 4, 5}


def test_threshold_uses_validation_video_false_alarm_constraint():
    validation = [
        {"label": 2, "max_drowsy_score": 0.50},
        {"label": 2, "max_drowsy_score": 0.90},
        {"label": 0, "max_drowsy_score": 0.10},
        {"label": 1, "max_drowsy_score": 0.50},
    ]

    threshold = choose_threshold(validation, max_false_video_rate=0.5)

    assert threshold == 0.5


def test_summary_reports_video_warning_misses_and_clip_start_time():
    data = {
        "y": np.asarray([0, 1, 2, 2]),
        "groups": np.asarray(["01", "01", "02", "02"]),
        "video_ids": np.asarray(["alert", "low", "drowsy1", "drowsy2"]),
        "durations": np.asarray([60, 60, 60, 60], dtype=np.float32),
        "starts": np.asarray([0, 0, 30, 0], dtype=np.float32),
    }
    indices = np.arange(4)
    probabilities = np.asarray([
        [0.90, 0.05, 0.05], [0.40, 0.50, 0.10],
        [0.10, 0.10, 0.80], [0.10, 0.10, 0.20],
    ], dtype=np.float32)

    rows = video_rows(data, indices, probabilities)
    report = summarize(data, indices, rows, threshold=0.70)

    assert report["drowsy_videos_missed"] == 1
    assert report["drowsy_videos_total"] == 2
    assert report["non_drowsy_videos_warned"] == 0
    assert report["median_first_warning_seconds_after_clip_start_on_detected_drowsy_videos"] == 30
