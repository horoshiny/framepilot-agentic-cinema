from .schemas import DepthLayout


def heuristic_depth_layout() -> DepthLayout:
    """Return deterministic, exclusive-friendly depth bands for any image."""

    return DepthLayout(
        foreground_occluders={
            "polygon": [
                {"x": 0, "y": 700},
                {"x": 1000, "y": 700},
                {"x": 1000, "y": 1000},
                {"x": 0, "y": 1000},
            ],
            "rationale": "Lower frame and edge detail is treated as the nearest occluding plane.",
        },
        primary_subject_midground={
            "polygon": [
                {"x": 220, "y": 220},
                {"x": 780, "y": 220},
                {"x": 820, "y": 700},
                {"x": 180, "y": 700},
            ],
            "rationale": "The central image band is reserved for the primary subject or midground action.",
        },
        background={
            "polygon": [
                {"x": 0, "y": 0},
                {"x": 1000, "y": 0},
                {"x": 1000, "y": 1000},
                {"x": 0, "y": 1000},
            ],
            "rationale": "The full frame is the backing region, with nearer polygons composited above it.",
        },
    )