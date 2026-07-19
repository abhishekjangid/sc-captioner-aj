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
    added_attributes: Optional[Mapping[str, Sequence[str]]] = None,
    removed_attributes: Optional[Mapping[str, Sequence[str]]] = None,
    added_relations: Optional[Sequence[Any]] = None,
    removed_relations: Optional[Sequence[Any]] = None
) -> ExplanationJson:
    """Build a structured explanation payload for SC-Captioner corrections."""
    _validate_caption_inputs(initial_caption, corrected_caption)

    normalized_added = _normalize_object_list(added_objects)
    normalized_removed = _normalize_object_list(removed_objects)
    normalized_yolo = _normalize_yolo_results(yolo_verification_results)
    normalized_added_attributes = _normalize_attribute_map(added_attributes)
    normalized_removed_attributes = _normalize_attribute_map(removed_attributes)
    normalized_added_relations = _normalize_relation_list(added_relations)
    normalized_removed_relations = _normalize_relation_list(removed_relations)

    verified_added_objects = [
        object_name for object_name in normalized_added if normalized_yolo.get(object_name, {}).get("verified", False)
    ]
    verified_removed_objects = [
        object_name for object_name in normalized_removed if normalized_yolo.get(object_name, {}).get("verified", False)
    ]
    hallucinations_removed = [
        object_name
        for object_name in normalized_removed
        if not normalized_yolo.get(object_name, {}).get("verified", False)
    ]
    confidence_score = _compute_confidence_score(
        verified_added_objects=verified_added_objects,
        hallucinations_removed=hallucinations_removed,
        added_objects=normalized_added,
        removed_objects=normalized_removed,
    )
    verification_results = _build_verification_results(normalized_yolo, normalized_added, normalized_removed)
    verified_objects = sorted(
        {
            result["object_name"]
            for result in verification_results
            if result["verified"]
        }
    )
    unverified_objects = sorted(
        {
            result["object_name"]
            for result in verification_results
            if not result["verified"]
        }
    )

    summary = _build_summary(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=normalized_added,
        removed_objects=normalized_removed,
        verified_added_objects=verified_added_objects,
        verified_removed_objects=verified_removed_objects,
        hallucinations_removed=hallucinations_removed,
        yolo_verification_results=normalized_yolo,
    )

    explanation = {
        "initial_caption": initial_caption.strip(),
        "corrected_caption": corrected_caption.strip(),
        "added_objects": normalized_added,
        "removed_objects": normalized_removed,
        "added_attributes": normalized_added_attributes,
        "removed_attributes": normalized_removed_attributes,
        "added_relations": normalized_added_relations,
        "removed_relations": normalized_removed_relations,
        "verified_added_objects": verified_added_objects,
        "verified_removed_objects": verified_removed_objects,
        "hallucinations_removed": hallucinations_removed,
        "verified_objects": verified_objects,
        "unverified_objects": unverified_objects,
        "verification_results": verification_results,
        "verification": {
            {
                "object": result["object_name"],
                "verified": result["verified"],
                "confidence": result["confidence"],
                "matched_label": result["matched_label"],
            }
            for result in verification_results
        },
        "changes": {
            "added_objects": normalized_added,
            "removed_objects": normalized_removed,
            "added_attributes": normalized_added_attributes,
            "removed_attributes": normalized_removed_attributes,
            "added_relations": normalized_added_relations,
            "removed_relations": normalized_removed_relations,
        },
        "confidence_score": confidence_score,
        "summary": summary,
        "human_readable_summary": summary,
    }
    LOGGER.debug("Generated SC-Captioner explanation JSON: %s", explanation)
    return explanation

def generate_explanation_text(
    initial_caption: str,
    corrected_caption: str,
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
    yolo_verification_results: Optional[Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = None,
    added_attributes: Optional[Mapping[str, Sequence[str]]] = None,
    removed_attributes: Optional[Mapping[str, Sequence[str]]] = None,
    added_relations: Optional[Sequence[Any]] = None,
    removed_relations: Optional[Sequence[Any]] = None,
) -> str:
    """Build a human-readable explanation for SC-Captioner corrections."""
    explanation = generate_explanation_json(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=added_objects,
        removed_objects=removed_objects,
        yolo_verification_results=yolo_verification_results,
        added_attributes=added_attributes,
        removed_attributes=removed_attributes,
        added_relations=added_relations,
        removed_relations=removed_relations,
    )

    normalized_yolo = _normalize_yolo_results(yolo_verification_results)
    verification_fragments = []
    for object_name in explanation["verified_added_objects"]:
        confidence = normalized_yolo.get(object_name, {}).get("confidence", 0.0)
        verification_fragments.append(f"{object_name} ({float(confidence):.2f})")


    verified_removed_fragments = []
    for object_name in explanation["verified_removed_objects"]:
        confidence = normalized_yolo.get(object_name, {}).get("confidence", 0.0)
        verified_removed_fragments.append(f"{object_name} ({float(confidence):.2f})")

    lines = [
        "SC-Captioner Explanation",
        f"Initial caption: {initial_caption.strip()}",
        f"Corrected caption: {corrected_caption.strip()}",
        "Added objects: {}".format(_format_object_list(explanation["added_objects"])),
        "Removed objects: {}".format(_format_object_list(explanation["removed_objects"])),
        "YOLO-verified corrections: {}".format(_format_object_list(verification_fragments)),
        "YOLO-verified removed objects: {}".format(_format_object_list(verified_removed_fragments)),
        "Hallucinations removed: {}".format(_format_object_list(explanation["hallucinations_removed"])),
        "Added attributes: {}".format(_format_change_map(explanation["changes"]["added_attributes"])),
        "Removed attributes: {}".format(_format_change_map(explanation["changes"]["removed_attributes"])),
        "Added relations: {}".format(_format_relation_list(explanation["changes"]["added_relations"])),
        "Removed relations: {}".format(_format_relation_list(explanation["changes"]["removed_relations"])),
        "Confidence score: {: .2f}".format(float(explanation["confidence_score"])),
        f"Summary: {explanation['summary']}",
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
    added_attributes: Optional[Mapping[str, Sequence[str]]] = None,
    removed_attributes: Optional[Mapping[str, Sequence[str]]] = None,
    added_relations: Optional[Sequence[Any]] = None,
    removed_relations: Optional[Sequence[Any]] = None,
) -> Dict[str, Union[ExplanationJson, str]]:
    """Return both JSON and human-readable explanation formats."""
    explanation_json = generate_explanation_json(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=added_objects,
        removed_objects=removed_objects,
        yolo_verification_results=yolo_verification_results,
        added_attributes=added_attributes,
        removed_attributes=removed_attributes,
        added_relations=added_relations,
        removed_relations=removed_relations,
    )
    explanation_text = generate_explanation_text(
        initial_caption=initial_caption,
        corrected_caption=corrected_caption,
        added_objects=added_objects,
        removed_objects=removed_objects,
        yolo_verification_results=yolo_verification_results,
        added_attributes=added_attributes,
        removed_attributes=removed_attributes,
        added_relations=added_relations,
        removed_relations=removed_relations,
    )
    return {
        "json": explanation_json,
        "human_readable": explanation_text,
    }

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

def _normalize_attribute_map(attributes: Optional[Mapping[str, Sequence[str]]]) -> Dict[str, List[str]]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping of object names to attribute sequences")

    normalized: Dict[str, List[str]] = {}
    for object_name, values in attributes.items():
        normalized_name = _normalize_object_list([object_name])
        if not normalized_name:
            continue
        normalized[normalized_name[0]] = _normalize_object_list(values)
    return normalized

def _normalize_relation_list(relations: Optional[Sequence[Any]]) -> List[str]:
    if relations is None:
        return []
    if not isinstance(relations, (list, tuple, set)):
        raise TypeError("relations must be a sequence")

    normalized: List[str] = []
    seen = set()
    for relation in relations:
        if isinstance(relation, str):
            relation_text = relation.strip().lower()
        elif isinstance(relation, (list, tuple)) and len(relation) >= 3:
            relation_text = " ".join(str(part).strip().lower() for part in relation[:3] if str(part).strip())
        else:
            relation_text = str(relation).strip().lower()

        if not relation_text or relation_text in seen:
            continue
        seen.add(relation_text)
        normalized.append(relation_text)
    return normalized

def _build_verification_results(
    normalized_yolo: Mapping[str, YoloResult],
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
) -> List[Dict[str, Any]]:
    verification_results: List[Dict[str, Any]] = []
    for object_name in added_objects:
        result = normalized_yolo.get(object_name, {})
        verification_results.append(
            {
                "object_name": object_name,
                "change_type": "added",
                "verified": bool(result.get("verified", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "matched_label": result.get("matched_label"),
            }
        )
    for object_name in removed_objects:
        result = normalized_yolo.get(object_name, {})
        verification_results.append(
            {
                "object_name": object_name,
                "change_type": "removed",
                "verified": bool(result.get("verified", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "matched_label": result.get("matched_label"),
            }
        )
    return verification_results

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
    hallucinations_removed: Sequence[str],
    yolo_verification_results: Mapping[str, YoloResult],
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
            "YOLO still detected the removed objects {}, so these removals should be verified carefully.".format(_format_object_list(verified_removed_objects))
        )

    if hallucinations_removed:
        summary_parts.append(
            "The removed objects {} are treated as hallucination removed from the initial caption.".format(
                _format_object_list(hallucinations_removed)
            )
        )

    unresolved_additions = [
        object_name for object_name in added_objects if object_name not in verified_added_objects
    ]
    if unresolved_additions:
        summary_parts.append(
            "The added objects {} were not verified by YOLO and should be reviewed if stricter validation is required.".format(
                _format_object_list(unresolved_additions)
            )
        )

    if initial_caption.strip() == corrected_caption.strip():
        summary_parts.append("The initial and corrected captions are textually identical.")

    return " ".join(summary_parts)

def _compute_confidence_score(
    verified_added_objects: Sequence[str],
    hallucinations_removed: Sequence[str],
    added_objects: Sequence[str],
    removed_objects: Sequence[str],
) -> float:
    total_corrections = len(added_objects) + len(removed_objects)
    if total_corrections <= 0:
        return 0.0

    confidence_score = len(verified_added_objects) + len(hallucinations_removed) / float(total_corrections)
    return round(confidence_score, 4)

def _format_object_list(objects: Iterable[Any]) -> str:
    items = [str(object_name) for object_name in objects if str(object_name)]
    return ", ".join(items) if items else "none"

def _format_change_map(values: Mapping[str, Sequence[str]]) -> str:
    if not values:
        return "none"
    return "; ".join(f"{key}: {', '.join(item for item in items) if items else 'none'}" for key, items in values.items())

def _format_relation_list(relations: Sequence[str]) -> str:
    return ", ".join(relations) if relations else "none"