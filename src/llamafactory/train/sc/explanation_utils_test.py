from llamafactory.train.sc.explanation_utils import generate_explanation
print(generate_explanation(
    initial_caption="a cat and a dog",
    corrected_caption="a cat",
    added_objects=[],
    removed_objects=["dog"],
    yolo_verification_results={"dog": {"verified": False, "confidence": 0.03}}
))