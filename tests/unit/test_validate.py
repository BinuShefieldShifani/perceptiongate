from perceptiongate.data.schema import Box, Frame
from perceptiongate.data.validate import ValidationReport, clean_frames

CLASSES = ["car", "pedestrian"]


def run(boxes: list[Box], limit: float = 1.0) -> tuple[list[Frame], ValidationReport]:
    return clean_frames([Frame("a.jpg", boxes=tuple(boxes))], CLASSES, 1280, 720, limit, "train")


def test_bad_boxes_are_dropped_and_counted() -> None:
    cleaned, report = run(
        [
            Box("car", 10, 10, 100, 100),  # good
            Box("car", 50, 50, 40, 40),  # inverted
            Box("car", 1200, 10, 1500, 100),  # outside image
            Box("car", 10, 10, 11, 50),  # 1 px wide
            Box("train", 10, 10, 100, 100),  # class we don't use
        ]
    )
    assert len(cleaned[0].boxes) == 1
    assert report.dropped["non_positive_size"] == 1
    assert report.dropped["outside_image"] == 1
    assert report.dropped["tiny"] == 1
    assert report.dropped["class_not_used"] == 1


def test_too_many_invalid_boxes_fails() -> None:
    _, report = run([Box("car", 50, 50, 40, 40), Box("car", 10, 10, 100, 100)], limit=0.1)
    assert not report.ok


def test_duplicate_frames_fail() -> None:
    f = Frame("a.jpg", boxes=(Box("car", 0, 0, 10, 10),))
    _, report = clean_frames([f, f], CLASSES, 1280, 720, 1.0, "val")
    assert any("duplicate" in e for e in report.errors)


def test_missing_class_is_a_warning() -> None:
    _, report = run([Box("car", 10, 10, 100, 100)])
    assert any("pedestrian" in w for w in report.warnings)
