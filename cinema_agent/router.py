import re
from typing import Literal

from .schemas import RoutingDecision


RoutingClass = Literal[
    "LOCAL_2_5D",
    "GENERATIVE_VIDEO_REQUIRED",
    "UNSUPPORTED",
]


_GENERATIVE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "character movement",
        (
            r"\bwalk(?:s|ed|ing)?\b",
            r"\bstroll(?:s|ed|ing)?\b",
            r"\bmarch(?:es|ed|ing)?\b",
            r"\brun(?:s|ning)?\b",
            r"\bjog(?:s|ged|ging)?\b",
            r"\b(?:take|takes|took|taking)\s+(?:a|one|another|two)?\s*steps?\b",
            r"\bsteps?\s+(?:forward|backward|back|toward|towards|away)\b",
            r"\bapproach(?:es|ed|ing)?\b",
            r"\b(?:raise|raises|raised|raising|lift|lifts|lifted|lifting)\b",
            r"\b(?:reach|reaches|reached|reaching)\b",
            r"\b(?:sit|sits|sat|sitting|stand|stands|stood|standing)\b",
            r"\b(?:kneel|kneels|knelt|kneeling|crouch|crouches|crouched|crouching)\b",
            r"\b(?:look|looks|looked|looking|gaze|gazes|gazed|gazing)\b",
            r"\b(?:nod|nods|nodded|nodding|wave|waves|waved|waving)\b",
            r"\b(?:grab|grabs|grabbed|grabbing|touch|touches|touched|touching)\b",
            r"\b(?:pick|picks|picked|picking)\s+up\b",
            r"\b(?:drop|drops|dropped|dropping|throw|throws|threw|throwing)\b",
            r"\bturn(?:s|ed|ing)?\s+(?:around|their|his|her|its)\b",
        ),
    ),
    (
        "door or gate movement",
        (
            r"\b(?:open|opens|opened|opening|close|closes|closed|closing)\s+"
            r"(?:the|a|an)?\s*(?:door|gate|window)\b",
            r"\b(?:door|gate|window)\s+(?:opens?|opened|opening|closes?|closed|closing)\b",
        ),
    ),
    (
        "object movement",
        (
            r"\b(?:swing|swings|swung|swinging|roll|rolls|rolled|rolling|"
            r"move|moves|moved|moving|bend|bends|bent|bending|rotate|rotates|"
            r"rotated|rotating|pulse|pulses|pulsed|pulsing|"
            r"pulsed|pulsing|glow|glows|glowed|glowing)\b",
        ),
    ),
    (
        "facial movement",
        (
            r"\b(?:blink|blinks|blinked|blinking|smile|smiles|smiled|smiling)\b",
            r"\b(?:frown|frowns|frowned|frowning|grimace|grimaces|winks?|"
            r"speaks?|whispers?|mouth)\b",
            r"\bfacial\s+(?:expression|movement|motion)\b",
        ),
    ),
    (
        "cloth or hair movement",
        (
            r"\b(?:cloth|clothing|fabric|garment|coat|cape|dress|shirt|sleeve|hair)\b"
            r".{0,50}\b(?:move|moves|moved|moving|flutter|flutters|fluttered|billow|"
            r"billows|rippl|sway)\w*\b",
            r"\b(?:move|moves|moved|moving|flutter|flutters|fluttered|billow|"
            r"billows|rippl|sway)\w*\b.{0,50}\b(?:cloth|clothing|fabric|garment|"
            r"coat|cape|dress|shirt|sleeve|hair)\b",
        ),
    ),
    (
        "object transformation",
        (
            r"\btransform(?:s|ed|ing)?\b",
            r"\bmorph(?:s|ed|ing)?\b",
            r"\b(?:turn|turns|turned|turning|change|changes|changed|changing)\s+"
            r"(?:into|to)\b",
            r"\b(?:become|becomes|became|becoming)\b",
            r"\b(?:melt|melts|melted|melting|dissolve|dissolves|dissolved|"
            r"dissolving)\b",
        ),
    ),
)

_UNSUPPORTED_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "unsupported physical spectacle",
        (
            r"\bteleport(?:s|ed|ing|ation)?\b",
            r"\bexplode(?:s|d|ing)?\b",
            r"\bexplosion\b",
            r"\bfly(?:s|ing)?\b",
            r"\blevitat(?:e|es|ed|ing)\b",
            r"\b(?:fight|fights|fought|fighting|combat)\b",
        ),
    ),
)

_LOCAL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "camera motion",
        (
            r"\bcamera\b",
            r"\bpush[- ]?in\b",
            r"\bpull[- ]?out\b",
            r"\bpan(?:s|ned|ning)?\b",
            r"\btilt(?:s|ed|ing)?\b",
            r"\btrack(?:s|ed|ing)?\b",
            r"\bzoom(?:s|ed|ing)?\b",
            r"\bdrift(?:s|ed|ing)?\b",
        ),
    ),
    (
        "2.5D depth and atmosphere",
        (
            r"\bparallax\b",
            r"\bdepth\b",
            r"\bfog\b",
            r"\bdust\b",
            r"\brain\b",
            r"\bembers?\b",
            r"\bhaze\b",
            r"\batmosphere\b",
            r"\blight\s+flicker(?:s|ed|ing)?\b",
            r"\bflicker(?:s|ed|ing)?\b",
        ),
    ),
)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _matched_labels(text: str, rules: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    return [label for label, patterns in rules if _matches(text, patterns)]


def route_scene_action(screenplay: str, mood: str = "") -> RoutingDecision:
    """Route a scene request to the capabilities FramePilot can safely represent.

    This is intentionally deterministic and runs before any Gemini or generation
    call. Semantic subject actions take precedence over local effects so a scene
    mentioning both, such as a person walking through fog, is not presented as
    fully supported by the browser renderer.
    """

    # Creative mood is intentionally excluded: phrases such as "running on
    # adrenaline" describe tone, not a requested on-screen action.
    _ = mood
    text = screenplay.strip()
    generative_actions = _matched_labels(text, _GENERATIVE_PATTERNS)
    unsupported_actions = _matched_labels(text, _UNSUPPORTED_PATTERNS)
    local_actions = _matched_labels(text, _LOCAL_PATTERNS)

    if unsupported_actions:
        return RoutingDecision(
            classification="UNSUPPORTED",
            rationale=(
                f"This request includes {', '.join(unsupported_actions)}, which is outside "
                "FramePilot's reliable local 2.5D controls. No paid generation call was made."
            ),
            matched_actions=unsupported_actions,
        )

    if generative_actions:
        return RoutingDecision(
            classification="GENERATIVE_VIDEO_REQUIRED",
            rationale=(
                f"This request includes {', '.join(generative_actions)}, which changes a "
                "semantic subject rather than the camera or atmosphere. A faithful result "
                "requires generative video approval. No paid generation API was called."
            ),
            matched_actions=generative_actions,
        )

    if local_actions:
        return RoutingDecision(
            classification="LOCAL_2_5D",
            rationale=(
                f"This request stays within browser-supported 2.5D controls: "
                f"{', '.join(local_actions)}."
            ),
            matched_actions=local_actions,
        )

    return RoutingDecision(
        classification="UNSUPPORTED",
        rationale=(
            "No reliable local 2.5D action was identified. FramePilot supports camera motion, "
            "parallax, zoom, pan, fog, dust, rain, embers, and light flicker."
        ),
        matched_actions=[],
    )