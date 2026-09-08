import base64
import io
import re

from PIL import Image, ImageFilter, ImageStat

from .router import route_scene_action
from .schemas import MotionCandidate, MotionPlan, ShotParameters


_CAMERA_REQUESTS = (
    (
        "push_in",
        (r"push(?:es|ed|ing)?[- ]?in", r"zoom(?:s|ed|ing)?\s+in", r"move closer"),
    ),
    (
        "pull_out",
        (r"pull(?:s|ed|ing)?[- ]?out", r"zoom(?:s|ed|ing)?\s+out", r"move away"),
    ),
    ("pan_left", (r"pan(?:s|ned|ning)?\s+left", r"track(?:s|ed|ing)?\s+left")),
    ("pan_right", (r"pan(?:s|ned|ning)?\s+right", r"track(?:s|ed|ing)?\s+right")),
)

def _screenplay_sentence(text: str, pattern: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return sentence.strip()[:500]
    return "No matching screenplay evidence was supplied."


def _image_observation(image_data_url: str | None) -> tuple[str, float, dict[str, str | float]]:
    if not image_data_url or "," not in image_data_url:
        return (
            "No uploaded storyboard image is available, so semantic subjects and object identities "
            "are intentionally unasserted.",
            0.0,
            {"orientation": "unknown", "tone": "unknown", "texture": "unknown"},
        )
    try:
        header, encoded = image_data_url.split(",", 1)
        if header.split(";")[0].replace("data:", "") not in {
            "image/png",
            "image/jpeg",
            "image/webp",
        }:
            raise ValueError("unsupported image type")
        image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
        width, height = image.size
        thumbnail = image.resize((min(64, width), min(64, height)))
        luminance = sum(ImageStat.Stat(thumbnail).mean) / 3
        contrast = sum(ImageStat.Stat(thumbnail).var) / 3
        edges = ImageStat.Stat(thumbnail.filter(ImageFilter.FIND_EDGES))
        edge_energy = sum(edges.mean) / 3
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        tone = "low-key" if luminance < 82 else "bright" if luminance > 175 else "mid-tone"
        texture = "textured" if edge_energy > 28 or contrast > 1800 else "visually restrained"
        return (
            f"Uploaded frame is {width}×{height} ({orientation}), {tone}, and {texture}; "
            "this pixel-level observation does not identify semantic subjects.",
            0.45,
            {
                "orientation": orientation,
                "tone": tone,
                "texture": texture,
                "luminance": luminance,
                "edge_energy": edge_energy,
            },
        )
    except (ValueError, OSError, base64.binascii.Error):
        return (
            "The uploaded frame could not be inspected reliably by the deterministic fallback; "
            "semantic subjects and object identities are intentionally unasserted.",
            0.0,
            {"orientation": "unknown", "tone": "unknown", "texture": "unknown"},
        )


def _camera_request(screenplay: str, mood: str) -> tuple[str, str]:
    camera_patterns = (
        ("pan_right", (r"\b(?:slow\s+)?slide(?:s|d|ing)?\s+right\b", r"\bpan(?:s|ned|ning)?\s+right\b")),
        ("pan_left", (r"\b(?:slow\s+)?slide(?:s|d|ing)?\s+left\b", r"\bpan(?:s|ned|ning)?\s+left\b")),
        *(_CAMERA_REQUESTS),
    )
    for sentence in _sentences(screenplay):
        if not re.search(
            r"\b(?:camera|shot|frame|slide|pan|track|push|pull|tilt|zoom|drift|move)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue
        for motion, patterns in camera_patterns:
            if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns):
                return motion, sentence[:500]
    mood_text = mood.lower()
    if re.search(r"\b(?:urgent|urgency|restless|agitated|panic)\b", mood_text):
        return "pan_right", "No explicit camera instruction was supplied in the screenplay."
    if re.search(r"\b(?:intimate|tender|quiet|contemplative)\b", mood_text):
        return "push_in", "No explicit camera instruction was supplied in the screenplay."
    return "drift", "No explicit camera movement was requested; a restrained drift is used."


def _camera_instruction_clause(evidence: str) -> str:
    if evidence.lower().startswith("no explicit camera"):
        return "No explicit camera instruction was supplied in the screenplay."
    clause = evidence.strip()
    camera_marker = re.search(r"\bcamera\s*:\s*", clause, flags=re.IGNORECASE)
    if camera_marker:
        clause = clause[camera_marker.end() :]
    else:
        clause = re.sub(r"^.*?\b(?:camera|shot|frame)\b\s*", "", clause, flags=re.IGNORECASE)
    clause = re.split(
        r"\s+(?:while|as|because|although)\s+",
        clause,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return clause.strip(" .,;:")[:180] or "Use only the bounded camera move identified in the evidence."


def _is_static_state_match(sentence: str, match: re.Match[str]) -> bool:
    prefix = sentence[: match.start()]
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", prefix.lower())
    return bool(words and words[-1] in _STATIC_STATE_VERBS)


_CHARACTER_ACTIONS = (
    "raise|raises|raised|raising|lift|lifts|lifted|lifting|"
    "lower|lowers|lowered|lowering|walk|walks|walked|walking|"
    "stroll|strolls|strolled|strolling|run|runs|ran|running|"
    "reach|reaches|reached|reaching|nod|nods|nodded|nodding|"
    "wave|waves|waved|waving|"
    "gesture|gestures|gestured|gesturing|kneel|kneels|knelt|kneeling"
)
_OBJECT_ACTIONS = (
    "rotate|rotates|rotated|rotating|swing|swings|swung|swinging|"
    "roll|rolls|rolled|rolling|move|moves|moved|moving|"
    "bend|bends|bent|bending|fall|falls|fell|falling|"
    "open|opens|opened|opening|close|closes|closed|closing|"
    "pulse|pulses|pulsed|pulsing|glow|glows|glowed|glowing|"
    "flicker|flickers|flickered|flickering|sway|sways|swayed|swaying"
)
_ENVIRONMENT_ACTIONS = (
    "drift|drifts|drifted|drifting|ripple|ripples|rippled|rippling|"
    "flow|flows|flowed|flowing|swirl|swirls|swirled|swirling|"
    "settle|settles|settled|settling|shimmer|shimmers|shimmered|shimmering|"
    "billow|billows|billowed|billowing|flicker|flickers|flickered|flickering|"
    "fall|falls|fell|falling|cross|crosses|crossed|crossing"
)
_ACTION_ADVERBS = {
    "a",
    "an",
    "the",
    "as",
    "at",
    "by",
    "from",
    "in",
    "into",
    "of",
    "on",
    "over",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "with",
    "and",
    "then",
    "slowly",
    "gently",
    "quietly",
    "subtly",
    "carefully",
    "deliberately",
}
_STATIC_STATE_VERBS = {
    "appear",
    "appears",
    "is",
    "are",
    "look",
    "looks",
    "remain",
    "remains",
    "seem",
    "seems",
    "stand",
    "stands",
    "stood",
    "sit",
    "sits",
    "sat",
    "was",
    "were",
}


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip(" \t\r\n-")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip(" \t\r\n-")
    ]


def _entity_before(sentence: str, match: re.Match[str]) -> str:
    prefix = re.split(r"\b(?:and|while|as|when|then)\b", sentence[: match.start()], flags=re.I)[-1]
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", prefix)
    while words and words[-1].lower() in _ACTION_ADVERBS:
        words.pop()
    while words and words[0].lower() in _ACTION_ADVERBS:
        words.pop(0)
    while words and words[0].lower() in {"a", "an", "the"}:
        words.pop(0)
    while words and words[0].lower() in {"one", "single", "controlled", "subtle", "brief"}:
        words.pop(0)
    return " ".join(words[-4:]).strip().lower() or "screenplay-named subject"


def _entity_after(sentence: str, match: re.Match[str]) -> str:
    suffix = re.split(
        r"\b(?:from|through|across|toward|towards|while|as|and|before|after|until)\b",
        sentence[match.end() :],
        maxsplit=1,
        flags=re.I,
    )[0]
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", suffix)
    while words and words[0].lower() in _ACTION_ADVERBS:
        words.pop(0)
    return " ".join(words[:4]).strip()


def _resolve_reference(label: str, fallback: str) -> str:
    normalized = label.strip().lower()
    if re.search(
        r"\b(?:it|its|they|them|their|he|him|his|she|her|this|that|these|those)\b",
        normalized,
    ) or re.match(r"^(?:same|former|aforementioned)\b", normalized):
        return fallback or "screenplay-named subject"
    if normalized in {"object", "subject", "element"}:
        return fallback or "screenplay-named subject"
    return label


def _intent_evidence(intent: str) -> str:
    return f"Creative intent: {intent.strip()[:400]}" if intent.strip() else "No creative intent was specified."


def _bounded_action(sentence: str, subject: str, verb: str, timing: str) -> str:
    action = sentence.strip()
    if timing and timing.lower() not in action.lower():
        action = f"{action} Timing: {timing}."
    return action[:300] or f"Bound the {verb} action for {subject} to the existing frame."


def _timing_clause(sentence: str) -> str:
    timing_patterns = (
        r"\b(?:in|during|throughout|over|after|before|until|for)\s+"
        r"(?:the\s+)?(?:final|first|last|middle|opening|closing|next)\s+"
        r"(?:third|half|moment|beat|seconds?|part)\b",
        r"\b(?:one|once|single|only|twice|repeatedly|briefly|continuously)\b",
        r"\bfor\s+\d+\s+seconds?\b",
    )
    for pattern in timing_patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def _is_detail_entity(entity: str) -> bool:
    return bool(
        re.search(
            r"\b(?:fabric|cloth|garment|hem|sleeve|coat|cape|dress|shirt|hair|"
            r"shadow|reflection|silhouette|surface)\b",
            entity,
            flags=re.I,
        )
    )


def _is_environment_entity(entity: str) -> bool:
    return bool(
        re.search(
            r"\b(?:air|ash|cloud|clouds|dust|fire|flame|fog|haze|light|rain|"
            r"reflection|river|smoke|snow|surface|water|wind|shadow|waves?|weather)\b",
            entity,
            flags=re.I,
        )
    )


def _valid_fallback_entity(label: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", label)
    if not words:
        return False
    normalized = [word.lower() for word in words]
    fragments = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
        "of", "on", "or", "the", "to", "with", "bend", "bending", "drift",
        "drifting", "fall", "falling", "flicker", "flickering", "flow",
        "flowing", "glow", "glowing", "move", "moving", "raise", "raising",
        "rotate", "rotating", "rotates", "run", "running", "slow", "slowly",
        "subtle", "subtly", "walk", "walking", "walks", "wave", "waving",
        "waves", "open", "opens", "opening",
    }
    if "screenplay-named" in normalized:
        return False
    if any(word in fragments for word in normalized):
        return False
    return not normalized[0].endswith("ly")


def _screenplay_candidate(
    *,
    label: str,
    action: str,
    bound: str,
    screenplay: str,
    evidence_pattern: str,
    observation: str,
    intent: str = "Creative intent was not specified separately.",
    screenplay_evidence: str | None = None,
) -> MotionCandidate:
    return MotionCandidate(
        label=label,
        action=action,
        bound=bound,
        visual_evidence=observation,
        screenplay_evidence=screenplay_evidence or _screenplay_sentence(screenplay, evidence_pattern),
        intent_evidence=intent,
        confidence=0.72,
        visual_confidence=0.0,
        grounding_source="screenplay",
        support="needs_confirmation",
    )


def _screenplay_grounded_motion(
    screenplay: str,
    observation: str,
    intent: str,
) -> tuple[
    list[MotionCandidate],
    list[MotionCandidate],
    list[MotionCandidate],
    list[MotionCandidate],
    list[str],
]:
    """Extract generic, screenplay-grounded actions without naming a scene."""

    character_pattern = re.compile(rf"\b(?:{_CHARACTER_ACTIONS})\b", re.IGNORECASE)
    object_pattern = re.compile(rf"\b(?:{_OBJECT_ACTIONS})\b", re.IGNORECASE)
    environment_pattern = re.compile(rf"\b(?:{_ENVIRONMENT_ACTIONS})\b", re.IGNORECASE)
    environment_nouns = re.compile(
        r"\b(?:air|ash|cloud|clouds|dust|fire|flame|fog|haze|light|rain|"
        r"reflection|river|smoke|snow|surface|water|wind|shadow|waves?|weather)\b",
        re.IGNORECASE,
    )
    characters: list[MotionCandidate] = []
    objects: list[MotionCandidate] = []
    environment: list[MotionCandidate] = []
    seen: set[tuple[str, str]] = set()
    last_character = ""
    last_object = ""
    last_environment = ""
    last_subject = ""

    def add_candidate(
        bucket: list[MotionCandidate],
        label: str,
        sentence: str,
        match: re.Match[str],
        bound: str,
        pattern: str,
    ) -> None:
        if not _valid_fallback_entity(label):
            return
        action = _bounded_action(sentence, label, match.group(0), _timing_clause(sentence))
        bucket_name = (
            "characters" if bucket is characters else "objects" if bucket is objects else "environment"
        )
        key = (bucket_name, action.lower())
        if key in seen:
            return
        seen.add(key)
        bucket.append(
            _screenplay_candidate(
                label=label,
                action=action,
                bound=bound,
                screenplay=screenplay,
                evidence_pattern=pattern,
                observation=observation,
                intent=_intent_evidence(intent),
                screenplay_evidence=sentence[:500],
            )
        )

    for sentence in _sentences(screenplay):
        if re.match(
            r"\s*(?:no|never|avoid|without|do\s+not|don't|preserve|keep|maintain)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        if character_match := character_pattern.search(sentence):
            subject = _resolve_reference(
                _entity_before(sentence, character_match),
                last_character or last_subject,
            )
            target_bucket = environment if _is_environment_entity(subject) else characters
            target_bound = (
                "Keep the force within the existing environment and composition; do not add a new source or subject."
                if target_bucket is environment
                else "Keep the named character within the existing identity and silhouette; do not add subjects or unlisted body actions."
            )
            add_candidate(
                target_bucket,
                subject,
                sentence,
                character_match,
                target_bound,
                character_pattern.pattern,
            )
            if subject != "screenplay-named subject":
                last_character = subject
                last_subject = subject
            introduced_object = _entity_after(sentence, character_match)
            if introduced_object and introduced_object.lower() not in {
                "the",
                "a",
                "an",
            }:
                last_object = introduced_object.lower()
        for match in object_pattern.finditer(sentence):
            if _is_static_state_match(sentence, match):
                continue
            subject = _resolve_reference(
                _entity_before(sentence, match),
                last_object or last_character or last_subject,
            )
            label = subject if subject != "screenplay-named subject" else _entity_after(sentence, match)
            if not label:
                label = "screenplay-named object"
            label = _resolve_reference(label, last_object or last_character or last_subject)
            if _is_environment_entity(label) or (
                environment_nouns.search(sentence) and environment_pattern.search(sentence)
            ):
                bucket = environment
            else:
                bucket = objects
            add_candidate(
                bucket,
                label,
                sentence,
                match,
                "Keep the named element in its existing position, identity, and shape; do not replace or deform it.",
                object_pattern.pattern,
            )
            if label != "screenplay-named subject":
                last_object = label
                last_subject = label
        for match in environment_pattern.finditer(sentence):
            if object_pattern.search(sentence) and not environment_nouns.search(sentence):
                continue
            subject = _resolve_reference(
                _entity_before(sentence, match),
                last_environment or last_subject,
            )
            label = subject if subject != "screenplay-named subject" else _entity_after(sentence, match)
            label = _resolve_reference(label, last_environment or last_subject)
            add_candidate(
                environment,
                label or "screenplay-named environment",
                sentence,
                match,
                "Keep the effect within the existing environment and composition; do not add a new source or subject.",
                environment_pattern.pattern,
            )
            if label != "screenplay-named subject":
                last_environment = label
                last_subject = label

    unsupported: list[MotionCandidate] = []
    routing = route_scene_action(screenplay, intent)
    if routing.classification == "UNSUPPORTED" and routing.matched_actions:
        for action_label in routing.matched_actions:
            unsupported.append(
                MotionCandidate(
                    label=action_label,
                    action=f"Reject unsupported action classified as {action_label}; do not generate it.",
                    bound="Exclude this action from approved motion and from the final Veo prompt.",
                    visual_evidence=observation,
                    screenplay_evidence=(
                        f"The screenplay triggered the unsupported-action rule: {action_label}."
                    ),
                    intent_evidence=_intent_evidence(intent),
                    confidence=0.0,
                    grounding_source="unsupported",
                    support="unsupported",
                )
            )

    return characters, objects, environment, unsupported, []


def _motion_plan(
    screenplay: str,
    mood: str,
    image_data_url: str | None,
) -> tuple[MotionPlan, dict[str, float | str]]:
    observation, image_confidence, image_profile = _image_observation(image_data_url)
    routing = route_scene_action(screenplay, mood)
    camera_motion, screenplay_evidence = _camera_request(screenplay, mood)
    characters, objects, environment, unsupported_actions, grounding_conflicts = (
        _screenplay_grounded_motion(screenplay, observation, mood)
    )
    camera_action = (
        f"Use a bounded {camera_motion.replace('_', ' ')} with no subject deformation. "
        f"Honor the camera instruction: {_camera_instruction_clause(screenplay_evidence)}"
    )
    camera = MotionCandidate(
        label="camera",
        action=camera_action[:300],
        bound=(
            "Stay within the compositor's overscan and the declared pan/zoom limits; "
            f"respect the {image_profile['orientation']} frame orientation."
        ),
        visual_evidence=observation,
        screenplay_evidence=screenplay_evidence,
        intent_evidence=_intent_evidence(mood),
        confidence=max(0.35, image_confidence),
        grounding_source="screenplay",
        support="supported",
    )

    prohibited = [
        "Do not move, deform, replace, or transform any character or object that is not grounded in the approved evidence.",
        "Preserve identity, subject count, silhouette, text, architecture, lighting direction, and composition.",
        "Do not add actions, abrupt motion, cuts, new subjects, object replacement, or major deformation absent from the approved evidence.",
    ]
    confirmation_required = False
    confirmation_question = None
    screenplay_grounded = [*characters, *objects, *environment]
    if routing.classification != "LOCAL_2_5D":
        requested = ", ".join(routing.matched_actions) or "the requested semantic action"
        prohibited.append(f"Keep {requested} out of the local compositor unless approved for generative video.")
    if screenplay_grounded or unsupported_actions or grounding_conflicts:
        confirmation_required = True
        if grounding_conflicts:
            confirmation_question = (
                "The storyboard and screenplay provide conflicting evidence for the listed motion. "
                "Review the conflict and approve only the bounded actions that match the intended scene."
            )
        elif unsupported_actions:
            confirmation_question = (
                "Some requested motion is unsupported and has been excluded. "
                "Approve only the bounded, evidence-grounded actions listed here?"
            )
        else:
            confirmation_question = (
                "The storyboard does not visually confirm all screenplay-grounded motion. "
                "Approve the bounded evidence-grounded actions listed here, with no unsupported additions?"
            )

    preserved_elements = [
        "Preserve all visible subjects, objects, silhouettes, text, and spatial relationships.",
        "Preserve the supplied frame's composition, subject identity, lighting direction, and color relationships.",
    ]
    continuity_patterns = (
        r"\b(?:preserve|preserves|preserved|preserving|keep|keeps|kept|maintain|"
        r"maintains|maintained|continuity|identity|silhouette|composition|"
        r"architecture|lighting|color|colour|subject count)\b",
    )
    for sentence in _sentences(screenplay):
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in continuity_patterns):
            if sentence not in preserved_elements:
                preserved_elements.append(sentence[:300])

    source_counts = {
        "visual": 0,
        "screenplay": len(screenplay_grounded),
        "visual_and_screenplay": 0,
        "unsupported": len(unsupported_actions),
    }
    plan = MotionPlan(
        grounding_summary=(
            f"GROUNDING SOURCES: visual={source_counts['visual']}, "
            f"screenplay={source_counts['screenplay']}, "
            f"visual_and_screenplay={source_counts['visual_and_screenplay']}, "
            f"unsupported={source_counts['unsupported']}. "
            f"{observation} {_intent_evidence(mood)} "
            "Screenplay-grounded actions remain bounded and confirmation-gated; unsupported "
            "actions are excluded from approved candidates."
        ),
        movable_characters=characters,
        movable_objects=objects,
        environmental_motion=environment,
        camera_movement=camera,
        preserved_elements=preserved_elements[:12],
        prohibited_changes=prohibited,
        unsupported_actions=unsupported_actions,
        grounding_conflicts=grounding_conflicts,
        confirmation_required=confirmation_required,
        confirmation_question=confirmation_question,
    )
    image_metrics = {
        "image_confidence": image_confidence,
        "has_image": bool(image_data_url),
        "camera_motion": camera_motion,
        "environment_count": len(environment),
        "image_profile": image_profile,
    }
    return plan, image_metrics


def demo_direction(
    screenplay: str,
    mood: str,
    image_data_url: str | None = None,
) -> tuple[ShotParameters, MotionPlan]:
    motion_plan, metrics = _motion_plan(screenplay, mood, image_data_url)
    camera_motion = str(metrics["camera_motion"])
    image_confidence = float(metrics["image_confidence"])
    has_image = bool(metrics["has_image"])
    image_profile = metrics["image_profile"]
    environment = [
        item.label.replace(" ", "_")
        for item in motion_plan.environmental_motion
        if item.support == "supported"
    ]
    zoom_end = {
        "push_in": 1.12,
        "pull_out": 1.0,
        "pan_left": 1.05,
        "pan_right": 1.05,
        "drift": 1.04,
    }[camera_motion]
    pan_x = -5 if camera_motion == "pan_left" else 5 if camera_motion == "pan_right" else 0
    pan_y = 0
    if has_image and image_confidence >= 0.45:
        parallax = 0.38 if image_profile["texture"] == "textured" else 0.3
        intensity = 0.2 if image_profile["tone"] == "low-key" else 0.28
        if camera_motion == "drift":
            if image_profile["orientation"] == "portrait":
                pan_y = -3
                zoom_end = 1.06
            elif image_profile["orientation"] == "landscape":
                pan_x = 2
                zoom_end = 1.04
    else:
        parallax = 0.22
        intensity = 0.18
    return (
        ShotParameters(
            duration_seconds=8,
            camera_motion=camera_motion,
            zoom_start=1.0,
            zoom_end=zoom_end,
            pan_x=pan_x,
            pan_y=pan_y,
            parallax_strength=parallax,
            motion_intensity=intensity,
            atmosphere=environment,
            transition="fade",
        ),
        motion_plan,
    )