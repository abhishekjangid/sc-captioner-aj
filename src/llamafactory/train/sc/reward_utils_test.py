from llamafactory.train.sc.reward_utils import (
    build_reward_explanation_report,
    compute_object_hallucination_penalty,
    compute_object_removal_reward,
)

def test_added_object_verified_by_yolo_skips_penalty() -> None:
    penalty = compute_object_hallucination_penalty(
        object_names=["laptop"],
        image_path="image.jpg",
        verify_object_fn=lambda _image_path, _object_name: {"verified": True, "confidence": 0.9},
        verification_penalty_reduction=1.0,
    )

    assert penalty == 0.0

def test_verification_disabled_preserves_original_behavior() -> None:
    penalty = compute_object_hallucination_penalty(
        object_names=["laptop"],
        image_path="image.jpg",
        verify_object_fn=None,
        verification_penalty_reduction=1.0,
    )
    reward = compute_object_removal_reward(
        object_names=["cup"],
        image_path="image.jpg",
        verify_object_fn=None,
        verified_removal_reward_reduction=1.0,
    )

    assert penalty == 0.25
    assert reward == 0.25

def test_added_object_unverified_applies_original_penalty() -> None:
    penalty = compute_object_hallucination_penalty(
        object_names=["laptop"],
        image_path="image.jpg",
        verify_object_fn=lambda _image_path, _object_name: {"verified": False, "confidence": 0.0},
        verification_penalty_reduction=1.0,
    )

    assert penalty == 0.25

def test_removed_object_verified_by_yolo_skips_reward() -> None:
    reward = compute_object_removal_reward(
        object_names=["cup"],
        image_path="image.jpg",
        verify_object_fn=lambda _image_path, _object_name: {"verified": True, "confidence": 0.85},
        verified_removal_reward_reduction=1.0,
    )

    assert reward == 0.0

def test_removed_object_unverified_applies_original_reward() -> None:
    reward = compute_object_removal_reward(
        object_names=["cup"],
        image_path="image.jpg",
        verify_object_fn=lambda _image_path, _object_name: {"verified": False, "confidence": 0.0},
        verified_removal_reward_reduction=1.0,
    )

    assert reward == 0.25

def test_missing_image_path_falls_back_safely() -> None:
    penalty = compute_object_hallucination_penalty(
        object_names=["laptop"],
        image_path=None,
        verify_object_fn=lambda _image_path, _object_name: {"verified": False, "confidence": 0.0},
        verification_penalty_reduction=1.0,
    )
    reward = compute_object_removal_reward(
        object_names=["cup"],
        image_path=None,
        verify_object_fn=lambda _image_path, _object_name: {"verified": False, "confidence": 0.0},
        verified_removal_reward_reduction=1.0,
    )

    assert penalty == 0.25
    assert reward == 0.25

    penalty = compute_object_hallucination_penalty(
        object_names=["laptop"],
        image_path="image.jpg",
        verify_object_fn=_raise,
        verification_penalty_reduction=1.0,
    )
    reward = compute_object_removal_reward(
        object_names=["cup"],
        image_path="image.jpg",
        verify_object_fn=_raise,
        verified_removal_reward_reduction=1.0,
    )

    assert penalty == 0.25
    assert reward == 0.25

def test_build_reward_explanation_report_returns_simple_schema() -> None:
    explanation = build_reward_explanation_report(
        initial_caption="a dog on a sofa",
        corrected_caption="a sofa",
        added_objects=[],
        removed_objects=["dog"],
        yolo_verification_results={"dog": {"verified": False, "confidence": 0.01, "matched_label": None}},
        removed_objects_missing_from_reference=["dog"],
    )

    assert set(explanation.keys()) == {
        "initial_caption",
        "corrected_caption",
        "changes",
        "verification",
        "reward_decisions",
        "confidence_score",
        "summary",
    }