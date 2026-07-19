import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

LOGGER = logging.getLogger(__name__)


class YoloVerificationError(RuntimeError):
    """Raised when image verification cannot be completed."""


class YoloObjectVerifier:
    """Lazy YOLOv8n verifier with per-image detection caching."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._device: Optional[str] = None
        self._cache: Dict[Path, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def verify(self, image_path: str, object_name: str) -> Dict[str, Union[float, bool]]:
        """Verify whether an object is detected in an image.

            Returns a dictionary with:
            {"verified": bool, "confidence": float}
        """
        normalized_object = self._normalize_object_name(object_name)
        if not normalized_object:
            raise ValueError("object_name must be a non-empty string")

        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Image not found: {image}")

        try:
            detections = self._get_detections(image)
        except Exception as exc:
            LOGGER.exception("YOLO verification failed for image=%s object=%s", image, normalized_object)
            if isinstance(exc, YoloVerificationError):
                raise
            raise YoloVerificationError(str(exc)) from exc

        best_confidence = 0.0
        for detection in detections:
            detected_name = detection["name"]
            if detected_name == normalized_object:
                best_confidence = max(best_confidence, detection["confidence"])

        verified = best_confidence >= self.confidence_threshold
        result = {"verified": verified, "confidence": float(best_confidence)}
        LOGGER.debug(
            "YOLO verification result image=%s object=%s verified=%s confidence=%.4f",
            image,
            normalized_object,
            verified,
            best_confidence,
        )
        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def _get_detections(self, image: Path) -> List[Dict[str, Any]]:
        with self._lock:
            cached = self._cache.get(image)
            if cached is not None:
                LOGGER.debug("YOLO cache hit for image=%s", image)
                return cached

        LOGGER.info("Running YOLO detection for image=%s", image)
        model = self._get_model()

        try:
            results = model.predict(source=str(image), device=self._device, verbose=False)
        except Exception as exc:
            raise YoloVerificationError(f"YOLO prediction failed for {image}: {exc}") from exc

        detections = self._parse_results(results)
        with self._lock:
            self._cache[image] = detections
        LOGGER.info("Cached %d YOLO detections for image=%s", len(detections), image)
        return detections

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model

            yolo_class = self._import_yolo()
            self._device = self._select_device()
            LOGGER.info("Loading YOLO model=%s device=%s", self.model_name, self._device)
            try:
                model = yolo_class(self.model_name)
            except Exception as exc:
                raise YoloVerificationError(f"Unable to load YOLO model {self.model_name}: {exc}") from exc
            self._model = model
            return self._model

    @staticmethod
    def _import_yolo() -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise YoloVerificationError(
                "ultralytics is required for YOLO verification. Install it with `pip install ultralytics`."
            ) from exc

        return YOLO

    @staticmethod
    def _select_device() -> str:
        try:
            import torch
        except ImportError:
            LOGGER.warning("torch is not available; falling back to CPU for YOLO verification")
            return "cpu"

        if torch.backends.mps.is_available():
            LOGGER.info("Using Apple Silicon MPS for YOLO verification")
            return "mps"

        LOGGER.info("Using CPU for YOLO verification")
        return "cpu"

    @staticmethod
    def _normalize_object_name(object_name: str) -> str:
        return object_name.strip().lower()

    @staticmethod
    def _parse_results(results: Sequence[Any]) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            classes = YoloObjectVerifier._tensor_to_list(getattr(boxes, "cls", []))
            confidences = YoloObjectVerifier._tensor_to_list(getattr(boxes, "conf", []))

            for class_id, confidence in zip(classes, confidences):
                name = names.get(int(class_id))
                if name is None:
                    continue
                detections.append(
                    {
                        "name": str(name).strip().lower(),
                        "confidence": float(confidence),
                    }
                )
        return detections

    @staticmethod
    def _tensor_to_list(values: Any) -> List[Any]:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            return list(values.tolist())
        return list(values)


_DEFAULT_VERIFIER = YoloObjectVerifier()


def verify_object(image_path: str, object_name: str) -> Dict[str, Union[float, bool]]:
    """Module-level helper for one-off object verification."""
    return _DEFAULT_VERIFIER.verify(image_path=image_path, object_name=object_name)