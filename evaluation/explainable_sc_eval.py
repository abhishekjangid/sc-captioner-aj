import argparse
import collections
import json
import os
import pickle
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llamafactory.train.sc.reward_utils import get_revision
from llamafactory.train.sc.yolo_verifier import YoloObjectVerifier

def _load_capture_class(dataset_name: str):
    if dataset_name == "cocoln500":
        from evaluate_cocoln500.capture import CAPTURE
        return CAPTURE
    if dataset_name == "docci500":
        from evaluate_docci500.capture import CAPTURE
        return CAPTURE
    raise ValueError("Unsupported dataset: {}".format(dataset_name))

def _dataset_paths(dataset_name: str) -> Dict[str, str]:
    if dataset_name == "cocoln500":
        return {
            "gt": "evaluate_cocoln500/test_cocoln_500_gt.json",
            "extra_objects": "evaluate_cocoln500/cocoln500_gpt_objects.jsonl",
            "extra_attributes": "evaluate_cocoln500/cocoln500_gpt_attributes.json",
            "gt_parsed": "evaluate_cocoln500/gt_parsed_ours.pkl",
        }
    if dataset_name == "docci500":
        return {
            "gt": "evaluate_docci500/test_docci_500_gt.json",
            "extra_objects": "evaluate_docci500/docci500_gpt_objects.jsonl",
            "extra_attributes": "evaluate_docci500/docci500_gpt_attributes.json",
            "gt_parsed": "evaluate_docci500/gt_parsed_ours.pkl",
        }
    raise ValueError("Unsupported dataset: {}".format(dataset_name))

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _load_extra_objects(path: str) -> Dict[str, List[str]]:
    extra_objects = {}
    for item in _load_jsonl(path):
        key = next(iter(item))
        extra_objects[key] = item[key]
    return extra_objects

def _load_extra_attributes(path: str) -> Dict[str, Dict[str, List[str]]]:
    return _load_json(path)

def _load_pickle_if_exists(path: str) -> Optional[Any]:
    if os.path.isfile(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    return None

def _build_gt_parsed(
    refs: Mapping[str, List[str]],
    evaluator: Any,
    prev_gt_parsed: Optional[Any],
) -> Any:
    if prev_gt_parsed is not None:
        return prev_gt_parsed
    gts = [(sample_key, gt) for sample_key, sample_gts in refs.items() for gt in sample_gts]
    num_chunk = 1
    try:
        import torch
        num_chunk = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
    except Exception:
        num_chunk = 1
    chunk_size = len(gts) // num_chunk if num_chunk > 0 else len(gts)
    partitioned_data = []
    start = 0
    for idx in range(num_chunk):
        end = start + chunk_size
        if idx < len(gts) % num_chunk:
            end += 1
        partitioned_data.append(gts[start:end])
        start = end
    if prev_gt_parsed is None:
        return evaluator.process_samples_multiprocessing(partitioned_data, desc="parsing gt")
    return prev_gt_parsed

def _normalize_prediction_item(item: Mapping[str, Any]) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    prediction = item.get("predict") or item.get("predict_turn2") or item.get("predict_turn1")
    if not isinstance(prediction, str):
        raise ValueError("Prediction row does not contain a supported prediction field.")
    label = item.get("label")
    rejected_text = item.get("rejected_text")
    image_path = item.get("image_path")
    explanation_report = item.get("explanation_report")
    return prediction, label, rejected_text, image_path, explanation_report

def _build_refs_and_preds(
    prediction_rows: Sequence[Mapping[str, Any]],
    gt_dict: Mapping[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    refs: Dict[str, List[str]] = {}
    preds: Dict[str, List[str]] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    caption_to_image = {caption: image_name for image_name, caption in gt_dict.items()}
    for item in prediction_rows:
        prediction, label, rejected_text, image_path, explanation_report = _normalize_prediction_item(item)
        if not isinstance(label, str):
            raise ValueError("Prediction row is missing `label`, which is required for evaluation alignment.")
        image_name = caption_to_image.get(label)
        if image_name is None:
            raise ValueError("Could not match prediction label to a ground-truth image.")
        refs[image_name] = [label]
        preds[image_name] = [prediction]
        metadata[image_name] = {
            "prediction": prediction,
            "label": label,
            "rejected_text": rejected_text,
            "image_path": image_path,
            "explanation_report": explanation_report,
        }
    return refs, preds, metadata

def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

def _compute_hallucination_rate(
    refs: Mapping[str, List[str]],
    preds: Mapping[str, List[str]],
    gt_parsed: Mapping[str, Sequence[Any]],
    cand_parsed: Mapping[str, Sequence[Any]],
    evaluator: Any,
) -> float:
    hallucinated_objects = 0
    predicted_objects = 0
    for sample_key in refs.keys():
        sample_gt_parsed = gt_parsed[sample_key][0]
        sample_cand_parsed = cand_parsed[sample_key][0]
        gt_objects = sample_gt_parsed[0]
        cand_objects = sample_cand_parsed[0]
        if len(cand_objects) == 0:
            continue
        predicted_objects += len(cand_objects)
        _, object_cand_match, _ = evaluator.compute_match(cand_objects, gt_objects)
        hallucinated_objects += max(len(cand_objects) - object_cand_match, 0)
    return _safe_divide(float(hallucinated_objects), float(predicted_objects))

def _build_verification_map(
    image_path: Optional[str],
    object_names: Iterable[str],
    verification_enabled: bool,
    verifier: Optional[YoloObjectVerifier],
) -> Dict[str, Dict[str, Any]]:
    verification_results: Dict[str, Dict[str, Any]] = {}
    if not verification_enabled or verifier is None or not image_path:
        for object_name in object_names:
            normalized = object_name.strip().lower()
            if normalized:
                verification_results[normalized] = {"verified": False, "confidence": 0.0}
        return verification_results
    for object_name in object_names:
        normalized = object_name.strip().lower()
        if not normalized or normalized in verification_results:
            continue
        try:
            verification_results[normalized] = verifier.verify(image_path=image_path, object_name=normalized)
        except Exception:
            verification_results[normalized] = {"verified": False, "confidence": 0.0}
    return verification_results

def _compute_verification_accuracy(
    metadata: Mapping[str, Dict[str, Any]],
    gt_parsed: Mapping[str, Sequence[Any]],
    cand_parsed: Mapping[str, Sequence[Any]],
    evaluator: Any,
    verification_enabled: bool,
    verifier: Optional[YoloObjectVerifier],
) -> float:
    matches = 0
    total = 0
    for sample_key, sample_meta in metadata.items():
        rejected_text = sample_meta.get("rejected_text")
        prediction = sample_meta.get("prediction")
        image_path = sample_meta.get("image_path")
        if not isinstance(rejected_text, str) or not isinstance(prediction, str):
            continue
        sample_gt_parsed = gt_parsed[sample_key][0]
        sample_cand_parsed = cand_parsed[sample_key][0]
        gt_objects = sample_gt_parsed[0]
        cand_objects = sample_cand_parsed[0]
        rejected_parsed = evaluator.sample_to_parse_results((sample_key, rejected_text))
        rejected_objects = rejected_parsed[1]
        rejected_attributes = rejected_parsed[2]
        rejected_relations = rejected_parsed[3]
        removed_objects, added_objects, _, _, _, _ = get_revision(
            set(rejected_objects),
            set(cand_objects),
            rejected_attributes,
            sample_cand_parsed[1],
            set(rejected_relations),
            set(sample_cand_parsed[2]),
            rejected_text,
            prediction,
            evaluator.text_encoder,
            stop_words=True,
        )
        verification_results = _build_verification_map(
            image_path=image_path,
            object_names=list(added_objects),
            verification_enabled=verification_enabled,
            verifier=verifier,
        )
        for object_name in added_objects:
            normalized = object_name.strip().lower()
            if not normalized:
                continue
            _, object_cand_match, _ = evaluator.compute_match([normalized], gt_objects)
            ground_truth_present = object_cand_match > 0
            verified = bool(verification_results.get(normalized, {}).get("verified", False))
            matches += int(verified == ground_truth_present)
            total += 1
    return _safe_divide(float(matches), float(total))

def evaluate_prediction_folder(
    dataset_name: str,
    prediction_folder: str,
    verification_enabled: bool,
    verification_model: str,
    verification_threshold: float,
) -> Dict[str, float]:
    dataset_paths = _dataset_paths(dataset_name)
    gt_dict = _load_json(dataset_paths["gt"])
    extra_objects = _load_extra_objects(dataset_paths["extra_objects"])
    extra_attributes = _load_extra_attributes(dataset_paths["extra_attributes"])
    prediction_rows = _load_jsonl(os.path.join(prediction_folder, "generated_predictions.jsonl"))
    refs, preds, metadata = _build_refs_and_preds(prediction_rows, gt_dict)
    capture_cls = _load_capture_class(dataset_name)
    evaluator = capture_cls()
    cand_parsed_cache = _load_pickle_if_exists(os.path.join(prediction_folder, "cand_parsed_ours_2.pkl"))
    gt_parsed_cache = _load_pickle_if_exists(dataset_paths["gt_parsed"])
    object_precision, object_recall, object_f1, _, _, _, cand_parsed = evaluator.compute_score(
        refs,
        preds,
        prev_gt_parsed=gt_parsed_cache,
        prev_cand_parsed=cand_parsed_cache,
        extra_objects=extra_objects,
        extra_attributes=extra_attributes,
        return_parse_results=True,
    )
    gt_parsed = _build_gt_parsed(refs, evaluator, gt_parsed_cache)
    verifier = None
    if verification_enabled:
        verifier = YoloObjectVerifier(
            model_name=verification_model,
            confidence_threshold=verification_threshold,
        )
    hallucination_rate = _compute_hallucination_rate(refs, preds, gt_parsed, cand_parsed, evaluator)
    verification_accuracy = _compute_verification_accuracy(
        metadata=metadata,
        gt_parsed=gt_parsed,
        cand_parsed=cand_parsed,
        evaluator=evaluator,
        verification_enabled=verification_enabled,
        verifier=verifier,
    )
    return {
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
        "hallucination_rate": hallucination_rate,
        "verification_accuracy": verification_accuracy,
    }

def build_result_table(results: Mapping[str, Mapping[str, float]]) -> str:
    headers = [
        "Variant",
        "Object Precision",
        "Object Recall",
        "Object F1",
        "Hallucination Rate",
        "Verification Accuracy",
    ]
    lines = ["| " + " | ".join(headers) + " |", " | " + " | ".join(["---"] * len(headers)) + " |"]
    for variant_name, metrics in results.items():
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                variant_name,
                metrics["object_precision"],
                metrics["object_recall"],
                metrics["object_f1"],
                metrics["hallucination_rate"],
                metrics["verification_accuracy"],
            )
        )
    return "\n".join(lines)

def save_evaluation_results(output_path: str, results: Mapping[str, Mapping[str, float]]) -> None:
    payload = {
        "results": results,
        "markdown_table": build_result_table(results),
        "table_schema": {
            "columns": [
                "Variant",
                "Object Precision",
                "Object Recall",
                "Object F1",
                "Hallucination Rate",
                "Verification Accuracy",
            ]
        },
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Explainable + Verification-Aware SC-Captioner outputs.")
    parser.add_argument("--dataset", choices=["cocoln500", "docci500"], required=True)
    parser.add_argument("--original-folder", required=True, help="Folder containing original SC-Captioner generated_predictions.jsonl")
    parser.add_argument("--verification-folder", required=True, help="Folder containing verification-aware generated_predictions.jsonl")
    parser.add_argument("--verification-model", type=str, default="yolov8n.pt")
    parser.add_argument("--verification-threshold", type=float, default=0.40)
    parser.add_argument("--output", default=None, help="Optional path to save JSON results.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    results = {
        "Original SC-Captioner": evaluate_prediction_folder(
            dataset_name=args.dataset,
            prediction_folder=args.original_folder,
            verification_enabled=False,
            verification_model=args.verification_model,
            verification_threshold=args.verification_threshold,
        ),
        "Verification-Aware SC-Captioner": evaluate_prediction_folder(
            dataset_name=args.dataset,
            prediction_folder=args.verification_folder,
            verification_enabled=True,
            verification_model=args.verification_model,
            verification_threshold=args.verification_threshold,
        ),
    }
    table = build_result_table(results)
    print(table)
    if args.output:
        save_evaluation_results(args.output, results)

if __name__ == "__main__":
    main()