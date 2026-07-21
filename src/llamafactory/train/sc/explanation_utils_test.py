from llamafactory.train.sc.explanation_utils import generate_explanation
def test_explanation_schema_and_removed_object_semantics() -> None:
    explanation = generate_explanation(
        initial_caption="a person with a cup",
        corrected_caption="a person with a laptop",
        added_objects=["laptop"],
        removed_objects=["cup"],
        yolo_verification_results={
            "laptop": {"verified": True, "confidence": 0.92, "matched_label": "laptop"},
            "cup": {"verified": False, "confidence": 0.03, "matched_label": None},
        },
        added_objects_missing_from_reference=["laptop"],
        removed_objects_missing_from_reference=["cup"],
        verification_penalty_reduction=1.0,
        verified_removal_reward_reduction=1.0,
    )

    assert explanation["changes"] == {
        "added_objects": ["laptop"],
        "removed_objects": ["cup"],
    }
    assert explanation["verification"] == {
        "verified_added_objects": ["laptop"],
        "unverified_added_objects": [],
        "verified_removed_objects": [],
        "unverified_removed_objects": ["cup"],
    }
    assert explanation["reward_decisions"]["added_object_decisions"][0]["reward_action"] == "hallucination_penalty_skipped"
    assert explanation["reward_decisions"]["removed_object_decisions"][0]["reward_action"] == "removal_reward_applied"
    assert explanation["confidence_score"] == 0.5


def test_explanation_verified_removed_object_is_detected_in_image() -> None:
    explanation = generate_explanation(
        initial_caption="a person with a cup",
        corrected_caption="a person",
        added_objects=[],
        removed_objects=["cup"],
        yolo_verification_results={
            "cup": {"verified": True, "confidence": 0.81, "matched_label": "cup"},
        },
        added_objects_missing_from_reference=[],
        removed_objects_missing_from_reference=["cup"],
        verified_removal_reward_reduction=1.0,
    )

    assert explanation["verification"]["verified_removed_objects"] == ["cup"]
    assert explanation["verification"]["unverified_removed_objects"] == []
    assert explanation["reward_decisions"]["removed_object_decisions"][0]["reward_action"] == "removal_reward_skipped"