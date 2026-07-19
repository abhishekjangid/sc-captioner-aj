from llamafactory.train.sc.reward_utils import build_reward_explanation_report
print(build_reward_explanation_report(
    initial_caption="a dog on a sofa",
    corrected_caption="a sofa",
    added_objects=[],
    removed_objects=["dog"],
    yolo_verification_results={"dog": {"verified": False, "confidence": 0.01}}
))