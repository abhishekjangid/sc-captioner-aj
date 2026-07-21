import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger(__name__)

YoloResult = Dict[str, Any]
ExplanationJson = Dict[str, Any]

def generate_explanation_json(
    initial_caption: str,
    corrected_caption: str,
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
    yolo_verification_results: Optional[Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = None,
    added_objects_missing_from_reference: Optional[Sequence[str]] = None,
    removed_objects_missing_from_reference: Optional[Sequence[str]] = None,
    verification_penalty_reduction: float = 1.0,
    verified_removal_reward_reduction: float = 1.0,
) -> ExplanationJson:
    """Build a structured explanation payload for SC-Captioner corrections."""
    _validate_caption_inputs(initial_caption, corrected_caption)

    normalized_added = _normalize_object_list(added_objects)
    normalized_removed = _normalize_object_list(removed_objects)
    normalized_yolo = _normalize_yolo_results(yolo_verification_results)
    normalized_added_missing = _normalize_object_list(
        added_objects_missing_from_reference if added_objects_missing_from_reference is not None else normalized_added
    )
    normalized_removed_missing = _normalize_object_list(
        removed_objects_missing_from_reference if removed_objects_missing_from_reference is not None else normalized_removed
    )
    verified_added_objects = [
        object_name
        for object_name in normalized_added_missing
        if normalized_yolo.get(object_name, {}).get("verified", False)
    ]
    unverified_added_objects = [
        object_name for object_name in normalized_added_missing if object_name not in verified_added_objects
    ]
    verified_removed_objects = [
        object_name
        for object_name in normalized_removed_missing
        if normalized_yolo.get(object_name, {}).get("verified", False)
    ]
    unverified_removed_objects = [
        object_name for object_name in normalized_removed_missing if object_name not in verified_removed_objects
    ]
    confidence_score = _compute_confidence_score(
        verified_added_objects=verified_added_objects,
        verified_removed_objects=verified_removed_objects,
        added_objects_missing_from_reference=normalized_added_missing,
        removed_objects_missing_from_reference=normalized_removed_missing,
    )
    added_object_decisions = _build_added_object_decisions(
        normalized_added,
        normalized_added_missing,
        normalized_yolo,
        verification_penalty_reduction,
    )
    removed_object_decisions = _build_removed_object_decisions(
        normalized_removed,
        normalized_removed_missing,
        normalized_yolo,
        verified_removal_reward_reduction,
    )

    summary = _build_summary(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=normalized_added,
        removed_objects=normalized_removed,
        verified_added_objects=verified_added_objects,
        verified_removed_objects=verified_removed_objects,
        unverified_added_objects=unverified_added_objects,
        unverified_removed_objects=unverified_removed_objects,
    )

    explanation = {
        "initial_caption": initial_caption.strip(),
        "corrected_caption": corrected_caption.strip(),
        "changes": {
            "added_objects": normalized_added,
            "removed_objects": normalized_removed,
        },
        "verification": {
            "verified_added_objects": verified_added_objects,
            "unverified_added_objects": unverified_added_objects,
            "verified_removed_objects": verified_removed_objects,
            "unverified_removed_objects": unverified_removed_objects,
        },
        "reward_decisions": {
            "added_object_decisions": added_object_decisions,
            "removed_object_decisions": removed_object_decisions,
        },
        "confidence_score": confidence_score,
        "summary": summary
    }
    LOGGER.debug("Generated SC-Captioner explanation JSON: %s", explanation)
    return explanation

def generate_explanation_text(
    initial_caption: str,
    corrected_caption: str,
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
    yolo_verification_results: Optional[Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = None,
    added_objects_missing_from_reference: Optional[Sequence[str]] = None,
    removed_objects_missing_from_reference: Optional[Sequence[str]] = None,
    verification_penalty_reduction: float = 1.0,
    verified_removal_reward_reduction: float = 1.0,
) -> str:
    """Build a human-readable explanation for SC-Captioner corrections."""
    explanation = generate_explanation_json(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=added_objects,
        removed_objects=removed_objects,
        yolo_verification_results=yolo_verification_results,
        added_objects_missing_from_reference=added_objects_missing_from_reference,
        removed_objects_missing_from_reference=removed_objects_missing_from_reference,
        verification_penalty_reduction=verification_penalty_reduction,
        verified_removal_reward_reduction=verified_removal_reward_reduction,
    )

    normalized_yolo = _normalize_yolo_results(yolo_verification_results)
    verification_fragments = []
    for object_name in explanation["verification"]["verified_added_objects"]:
        confidence = normalized_yolo.get(object_name, {}).get("confidence", 0.0)
        verification_fragments.append(f"{object_name} ({float(confidence):.2f})")

    removed_verification_fragments = []
    for object_name in explanation["verification"]["verified_removed_objects"]:
        confidence = normalized_yolo.get(object_name, {}).get("confidence", 0.0)
        removed_verification_fragments.append(f"{object_name} ({float(confidence):.2f})")


    lines = [
        "SC-Captioner Explanation",
        f"Initial caption: {initial_caption.strip()}",
        f"Corrected caption: {corrected_caption.strip()}",
        "Added objects: {}".format(_format_object_list(explanation["changes"]["added_objects"])),
        "Removed objects: {}".format(_format_object_list(explanation["changes"]["removed_objects"])),
        "YOLO-verified added objects: {}".format(_format_object_list(verification_fragments)),
        "YOLO-unverified added objects: {}".format(_format_object_list(explanation["verification"]["unverified_added_objects"])),
        "YOLO-verified removed objects: {}".format(_format_object_list(removed_verification_fragments)),
        "YOLO-unverified removed objects: {}".format(_format_object_list(explanation["verification"]["unverified_removed_objects"])),
        "Confidence score: {:.2f}".format(float(explanation["confidence_score"])),
        f"Summary: {explanation['summary']}",
    ]
    human_readable = "\n".join(lines)
    LOGGER.debug("Generated SC-Captioner explanation text: %s", human_readable)
    return human_readable

def generate_explanation(
    initial_caption: str,
    corrected_caption: str,
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
    yolo_verification_results: Optional[Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = None,
    added_objects_missing_from_reference: Optional[Sequence[str]] = None,
    removed_objects_missing_from_reference: Optional[Sequence[str]] = None,
    verification_penalty_reduction: float = 1.0,
    verified_removal_reward_reduction: float = 1.0,
) -> ExplanationJson:
    """Return the structured explaination report used by SC-captioner."""
    return generate_explanation_json(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=added_objects,
        removed_objects=removed_objects,
        yolo_verification_results=yolo_verification_results,
        added_objects_missing_from_reference=added_objects_missing_from_reference,
        removed_objects_missing_from_reference=removed_objects_missing_from_reference,
        verification_penalty_reduction=verification_penalty_reduction,
        verified_removal_reward_reduction=verified_removal_reward_reduction,
    )

def explanation_to_json_string(explanation: Mapping[str, Any], indent: int = 2) -> str:
    """Serialize an explanation payload to a JSON string."""
    return json.dumps(dict(explanation), indent=indent, ensure_ascii=True)

def _validate_caption_inputs(initial_caption: str, corrected_caption: str) -> None:
    if not isinstance(initial_caption, str) or not initial_caption.strip():
        raise ValueError("initial_caption must be a non-empty string")

    if not isinstance(corrected_caption, str) or not corrected_caption.strip():
        raise ValueError("corrected_caption must be a non-empty string")

def _normalize_object_list(objects: Sequence[str]) -> List[str]:
    if objects is None:
        return []

    if not isinstance(objects, (list, tuple, set)):
        raise TypeError("objects must be a sequence of strings")

    normalized: List[str] = []
    seen = set()
    for object_name in objects:
        if not isinstance(object_name, str):
            raise TypeError("each object name must be a string")

        candidate = object_name.strip().lower()
        if not candidate or candidate in seen:
            continue
        
        seen.add(candidate)
        normalized.append(candidate)

    return normalized

def _normalize_yolo_results(
    yolo_verification_results: Optional[Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]]]
) -> Dict[str, YoloResult]:
    if yolo_verification_results is None:
        return {}

    if isinstance(yolo_verification_results, Mapping):
        items = yolo_verification_results.items()
    elif isinstance(yolo_verification_results, Sequence):
        items = _sequence_to_yolo_items(yolo_verification_results)
    else:
        raise TypeError("yolo_verification_results must be a mapping or a sequence of mappings")

    normalized: Dict[str, YoloResult] = {}
    for object_name, result in items:
        if not isinstance(object_name, str):
            raise TypeError("YOLO result object names must be strings")
        if not isinstance(result, Mapping):
            raise TypeError("YOLO verification results must be mappings")

        normalized_name = object_name.strip().lower()
        if not normalized_name:
            continue

        verified = bool(result.get("verified", False))
        confidence = float(result.get("confidence", 0.0))
        matched_label = result.get("matched_label")
        normalized[normalized_name] = {
            "verified": verified,
            "confidence": confidence,
            "matched_label": matched_label.strip().lower() if isinstance(matched_label, str) and matched_label.strip() else None,
        }

    return normalized

def _sequence_to_yolo_items(
    yolo_verification_results: Sequence[Mapping[str, Any]]
) -> List[Tuple[str, Mapping[str, Any]]]:
    items: List[Tuple[str, Mapping[str, Any]]] = []
    for result in yolo_verification_results:
        if not isinstance(result, Mapping):
            raise TypeError("YOLO verification sequence entries must be mappings")

        object_name = result.get("object_name") or result.get("name")
        if not isinstance(object_name, str):
            raise TypeError("YOLO verification sequence entries must include a string object_name")

        items.append((object_name, result))

    return items

def _build_summary(
    initial_caption: str,
    corrected_caption: str,
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
    verified_added_objects: Sequence[str],
    verified_removed_objects: Sequence[str],
    unverified_added_objects: Sequence[str],
    unverified_removed_objects: Sequence[str],
) -> str:
    summary_parts = [
        "The caption was revised to better match the image content.",
        f"{len(added_objects)} object(s) were added and {len(removed_objects)} object(s) were removed.",
    ]

    if verified_added_objects:
        summary_parts.append(
            "YOLO verified the added objects {}.".format(_format_object_list(verified_added_objects))
        )

    if verified_removed_objects:
        summary_parts.append(
            "YOLO detected the removed objects {}, so their removal reward was reduced or skipped.".format(
                _format_object_list(verified_removed_objects)
            )
        )

    if unverified_removed_objects:
        summary_parts.append(
            "YOLO did not detect the removed objects {}, so their removal reward was applied.".format(
                _format_object_list(unverified_removed_objects)
            )
        )

    if unverified_added_objects:
        summary_parts.append(
            "The added objects {} were not verified by YOLO and should be reviewed if stricter validation is required.".format(
                _format_object_list(unverified_added_objects)
            )
        )

    if initial_caption.strip() == corrected_caption.strip():
        summary_parts.append("The initial and corrected captions are textually identical.")

    return " ".join(summary_parts)

def _compute_confidence_score(
    verified_added_objects: Sequence[str],
    verified_removed_objects: Sequence[str],
    added_objects_missing_from_reference: Sequence[str],
    removed_objects_missing_from_reference: Sequence[str],
) -> float:
    total_verification_candidates = len(added_objects_missing_from_reference) + len(removed_objects_missing_from_reference)
    if total_verification_candidates <= 0:
        return 0.0

    confidence_score = (
        len(verified_added_objects) + len(verified_removed_objects)
    ) / float(total_verification_candidates)
    return round(confidence_score, 4)

def _build_added_object_decisions(
    added_objects: Sequence[str],
    added_objects_missing_from_reference: Sequence[str],
    normalized_yolo: Mapping[str, YoloResult],
    verification_penalty_reduction: float,
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    missing_set = set(added_objects_missing_from_reference)
    for object_name in added_objects:
        in_reference = object_name not in missing_set
        result = normalized_yolo.get(object_name, {})
        yolo_verified = bool(result.get("verified", False)) if not in_reference else False
        if in_reference:
            reward_action = "correctness_bonus_kept"
            reason = "Object was added and is already supported by the reference caption."
        elif yolo_verified:
            reward_action = (
                "hallucination_penalty_skipped"
                if verification_penalty_reduction >= 1.0
                else "hallucination_penalty_reduced"
            )
            reason = "Object was not present in reference but YOLO detected it in the image."
        else:
            reward_action = "hallucination_penalty_applied"
            reason = "Object was not present in reference and YOLO did not detect it in the image."
        decisions.append(
            {
                "object": object_name,
                "change_type": "added",
                "in_reference": in_reference,
                "yolo_verified": yolo_verified,
                "reward_action": reward_action,
                "reason": reason,
            }
        )
    return decisions

def _build_removed_object_decisions(
    removed_objects: Sequence[str],
    removed_objects_missing_from_reference: Sequence[str],
    normalized_yolo: Mapping[str, YoloResult],
    verified_removal_reward_reduction: float,
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    missing_set = set(removed_objects_missing_from_reference)
    for object_name in removed_objects:
        in_reference = object_name not in missing_set
        result = normalized_yolo.get(object_name, {})
        yolo_verified = bool(result.get("verified", False)) if not in_reference else False
        if in_reference:
            reward_action = "wrong_removal_penalty_kept"
            reason = "Object was removed but is still supported by the reference caption."
        elif yolo_verified:
            reward_action = (
                "removal_reward_skipped"
                if verified_removal_reward_reduction >= 1.0
                else "removal_reward_reduced"
            )
            reason = "Object was removed from the caption but YOLO detected it in the image."
        else:
            reward_action = "removal_reward_applied"
            reason = "Object was removed from the caption and YOLO did not detect it in the image."
        decisions.append(
            {
                "object": object_name,
                "change_type": "removed",
                "in_reference": in_reference,
                "yolo_verified": yolo_verified,
                "reward_action": reward_action,
                "reason": reason,
            }
        )
    return decisions
