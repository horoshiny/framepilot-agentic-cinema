from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

from .schemas import CameraGrammar, ShotParameters, ShotSignature


@dataclass(frozen=True)
class CompiledShot:
    shot: ShotParameters
    grammar: CameraGrammar
    signature: ShotSignature
    adjustment: str | None = None


@dataclass(frozen=True)
class _PreviousShot:
    intent_key: str
    signature: ShotSignature


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _intent_key(screenplay: str, mood: str) -> str:
    normalized = f"{mood.strip().lower()}\n{screenplay.strip().lower()}"
    return sha256(normalized.encode("utf-8")).hexdigest()


def _pan_direction(screenplay: str, mood: str) -> str:
    text = f"{screenplay} {mood}".lower()
    left_score = sum(text.count(token) for token in ("left", "west", "counterclockwise"))
    right_score = sum(text.count(token) for token in ("right", "east", "clockwise"))
    if left_score > right_score:
        return "left"
    if right_score > left_score:
        return "right"
    digest = sha256(text.encode("utf-8")).digest()
    return "left" if digest[0] % 2 else "right"


def select_camera_grammar(screenplay: str, mood: str) -> CameraGrammar:
    text = f"{mood} {screenplay}".lower()
    categories = (
        ("urgency", ("urgent", "urgency", "hurry", "frantic", "panic", "chase", "deadline")),
        ("intimacy", ("intimate", "tender", "vulnerable", "affection", "confession", "close-up")),
        ("awe", ("awe", "wonder", "majestic", "vast", "grandeur", "breathtaking", "reveal")),
        ("dread", ("dread", "uneasy", "ominous", "haunted", "menace", "fear", "suspense")),
        ("mystery", ("mystery", "mysterious", "unknown", "enigmatic", "secret", "strange", "clue")),
    )
    for grammar, tokens in categories:
        if any(token in text for token in tokens):
            return grammar
    return "balanced"


def _with_profile(shot: ShotParameters, grammar: CameraGrammar, pan_direction: str) -> ShotParameters:
    if grammar == "balanced":
        return shot

    sign = -1 if pan_direction == "left" else 1
    profiles = {
        "awe": {
            "camera_motion": "pan_left" if sign < 0 else "pan_right",
            "duration_seconds": 12.0,
            "zoom_start": 1.0,
            "zoom_end": 1.04,
            "pan_x": 14 * sign,
            "pan_y": 0,
            "parallax_strength": 0.86,
            "motion_intensity": 0.3,
            "atmosphere": [effect for effect in shot.atmosphere if effect != "light_flicker"],
        },
        "dread": {
            "camera_motion": "pull_out",
            "duration_seconds": 8.5,
            "zoom_start": 1.14,
            "zoom_end": 1.02,
            "pan_x": -8 * sign,
            "pan_y": 1,
            "parallax_strength": 0.42,
            "motion_intensity": 0.3,
            "atmosphere": ["light_flicker"],
        },
        "urgency": {
            "camera_motion": "pan_left" if sign < 0 else "pan_right",
            "duration_seconds": 4.5,
            "zoom_start": 1.0,
            "zoom_end": 1.2,
            "pan_x": 17 * sign,
            "pan_y": 0,
            "parallax_strength": 0.72,
            "motion_intensity": 0.92,
            "atmosphere": list(shot.atmosphere),
        },
        "intimacy": {
            "camera_motion": "push_in",
            "duration_seconds": 8.0,
            "zoom_start": 1.0,
            "zoom_end": 1.08,
            "pan_x": 0,
            "pan_y": 0,
            "parallax_strength": 0.16,
            "motion_intensity": 0.22,
            "atmosphere": [],
        },
        "mystery": {
            "camera_motion": "drift",
            "duration_seconds": 9.5,
            "zoom_start": 1.02,
            "zoom_end": 1.1,
            "pan_x": 9 * sign,
            "pan_y": -3,
            "parallax_strength": 0.56,
            "motion_intensity": 0.45,
            "atmosphere": ["fog"],
        },
    }
    return shot.model_copy(update=profiles[grammar])


def _signature(shot: ShotParameters) -> ShotSignature:
    if shot.pan_x < -0.5:
        pan_direction = "left"
    elif shot.pan_x > 0.5:
        pan_direction = "right"
    elif shot.camera_motion == "pan_left":
        pan_direction = "left"
    elif shot.camera_motion == "pan_right":
        pan_direction = "right"
    else:
        pan_direction = "hold"

    if shot.zoom_end > shot.zoom_start + 0.005:
        zoom_direction = "in"
    elif shot.zoom_end < shot.zoom_start - 0.005:
        zoom_direction = "out"
    else:
        zoom_direction = "hold"

    return ShotSignature(
        motion_type=shot.camera_motion,
        pan_direction=pan_direction,
        zoom_direction=zoom_direction,
        duration_seconds=round(shot.duration_seconds, 2),
        parallax_strength=round(shot.parallax_strength, 2),
        atmosphere=list(shot.atmosphere),
    )


def signatures_nearly_identical(first: ShotSignature, second: ShotSignature) -> bool:
    return (
        first.motion_type == second.motion_type
        and first.pan_direction == second.pan_direction
        and first.zoom_direction == second.zoom_direction
        and abs(first.duration_seconds - second.duration_seconds) <= 0.8
        and abs(first.parallax_strength - second.parallax_strength) <= 0.1
        and set(first.atmosphere) == set(second.atmosphere)
    )


def _force_distinct(
    shot: ShotParameters,
    grammar: CameraGrammar,
    previous: ShotSignature,
    pan_direction: str,
) -> tuple[ShotParameters, str]:
    sign = -1 if pan_direction == "left" else 1
    if previous.pan_direction == pan_direction:
        sign *= -1
    if grammar == "awe":
        updated = shot.model_copy(
            update={
                "camera_motion": "pan_left" if sign < 0 else "pan_right",
                "pan_x": 15 * sign,
                "zoom_start": 1.0,
                "zoom_end": 1.05,
                "duration_seconds": 13.0,
                "parallax_strength": 0.9,
            }
        )
        reason = "widened the lateral reveal and reversed its travel direction"
    elif grammar == "dread":
        updated = shot.model_copy(
            update={
                "camera_motion": "pull_out",
                "pan_x": 8 * sign,
                "zoom_start": 1.15,
                "zoom_end": 1.0,
                "duration_seconds": 7.5,
                "parallax_strength": 0.38,
                "atmosphere": ["light_flicker"],
            }
        )
        reason = "paired a pull-back with an opposing drift and restrained flicker"
    elif grammar == "urgency":
        updated = shot.model_copy(
            update={
                "camera_motion": "pan_left" if sign < 0 else "pan_right",
                "pan_x": 18 * sign,
                "zoom_start": 1.0,
                "zoom_end": 1.22,
                "duration_seconds": 4.0,
                "motion_intensity": 0.96,
            }
        )
        reason = "increased pan travel and pace for a sharper urgency beat"
    elif grammar == "intimacy":
        updated = shot.model_copy(
            update={
                "camera_motion": "push_in",
                "pan_x": 0,
                "zoom_start": 1.0,
                "zoom_end": 1.1,
                "duration_seconds": 8.0,
                "parallax_strength": 0.12,
            }
        )
        reason = "softened the frame into a distinct, low-parallax push-in"
    elif grammar == "mystery":
        updated = shot.model_copy(
            update={
                "camera_motion": "drift",
                "pan_x": 10 * sign,
                "pan_y": -4,
                "zoom_start": 1.02,
                "zoom_end": 1.11,
                "duration_seconds": 10.0,
                "parallax_strength": 0.6,
            }
        )
        reason = "introduced asymmetric drift and moderate parallax for mystery"
    else:
        updated = shot.model_copy(
            update={
                "camera_motion": "pan_left" if sign < 0 else "pan_right",
                "pan_x": 10 * sign,
                "duration_seconds": _clamp(shot.duration_seconds + 1.0, 3, 20),
                "motion_intensity": _clamp(shot.motion_intensity + 0.1, 0, 1),
            }
        )
        reason = "varied the balanced camera path to avoid repeating the previous shot"
    return updated, reason


class ShotDiversityGuard:
    """Keep recent per-client signatures without retaining screenplay or image content."""

    def __init__(self, max_clients: int = 128):
        self.max_clients = max_clients
        self._previous: OrderedDict[str, _PreviousShot] = OrderedDict()
        self._lock = RLock()

    def compile(
        self,
        shot: ShotParameters,
        screenplay: str,
        mood: str,
        client_key: str,
        *,
        force_distinct: bool = False,
    ) -> CompiledShot:
        grammar = select_camera_grammar(screenplay, mood)
        pan_direction = _pan_direction(screenplay, mood)
        compiled = _with_profile(shot, grammar, pan_direction)
        signature = _signature(compiled)
        intent_key = _intent_key(screenplay, mood)
        adjustment = None

        with self._lock:
            previous = self._previous.get(client_key)
            if previous and (force_distinct or previous.intent_key != intent_key):
                if signatures_nearly_identical(previous.signature, signature):
                    compiled, reason = _force_distinct(
                        compiled,
                        grammar,
                        previous.signature,
                        pan_direction,
                    )
                    signature = _signature(compiled)
                    adjustment = f"Diversity guard {reason}."
            self._previous[client_key] = _PreviousShot(intent_key, signature)
            self._previous.move_to_end(client_key)
            while len(self._previous) > self.max_clients:
                self._previous.popitem(last=False)

        return CompiledShot(compiled, grammar, signature, adjustment)


def compile_shot(
    shot: ShotParameters,
    screenplay: str,
    mood: str,
    client_key: str = "local",
    *,
    force_distinct: bool = False,
) -> CompiledShot:
    return ShotDiversityGuard().compile(
        shot,
        screenplay,
        mood,
        client_key,
        force_distinct=force_distinct,
    )


def signature_text(signature: ShotSignature) -> str:
    atmosphere = "+".join(signature.atmosphere) if signature.atmosphere else "none"
    return (
        f"{signature.motion_type.replace('_', ' ')} · pan {signature.pan_direction} · "
        f"zoom {signature.zoom_direction} · {signature.duration_seconds:g}s · "
        f"parallax {signature.parallax_strength:.0%} · fx {atmosphere}"
    )