import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from nltk.stem import WordNetLemmatizer
except ImportError:
    WordNetLemmatizer = None  # type: ignore

LOGGER = logging.getLogger(__name__)
_LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None


class YoloVerificationError(RuntimeError):
    """Raised when image verification cannot be completed."""


class YoloObjectVerifier:
    """Lazy YOLO object verifier with per-image detection caching."""

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.25,
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.allow_download = allow_download
        self._model = None
        self._device: Optional[str] = None
        self._cache: Dict[Path, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def verify(self, image_path: str, object_name: str) -> Dict[str, Any]:
        """Verify whether an object is detected in an image.

            Returns a dictionary with:
            {"verified": bool, "confidence": float}
        """
        normalized_object = self._normalize_object_name(object_name)
        if not normalized_object:
            raise ValueError("object_name must be a non-empty string")

        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            LOGGER.warning("YOLO verification skipped because image was not found: %s", image)
            return self._empty_result(error=f"Image not found: {image}")

        try:
            detections = self._get_detections(image)
        except Exception as exc:
            LOGGER.exception("YOLO verification failed for image=%s object=%s", image, normalized_object)
            if isinstance(exc, YoloVerificationError):
                return self._empty_result(error=str(exc))
            return self._empty_result(error=str(exc))
        
        LOGGER.info(
            "YOLO detections for image=%s %s",
            image,
            [(detection["name"], round(float(detection["confidence"]), 2)) for detection in detections]
        )

        best_confidence = 0.0
        matched_label: Optional[str] = None
        for detection in detections:
            detected_name = detection["name"]
            if self._is_matching_label(normalized_object, detected_name):
                best_confidence = max(best_confidence, detection["confidence"])
                if detection["confidence"] == best_confidence:
                    matched_label = detected_name

        verified = best_confidence >= self.confidence_threshold
        result = {
            "verified": verified, 
            "confidence": float(best_confidence),
            "matched_label": matched_label,
            "error": None
        }
        LOGGER.debug(
            "YOLO verification result image=%s object=%s verified=%s confidence=%.4f",
            image,
            normalized_object,
            verified,
            best_confidence,
        )
        return result
    

    def verify_object(self, image_path: str, object_name: str) -> Dict[str, Any]:
        """Compatibility wrapper matching the expected verifier interface."""
        return self.verify(image_path=image_path, object_name=object_name)

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
            model_path = Path(self.model_name).expanduser()
            if not self.allow_download and not model_path.is_file():
                raise YoloVerificationError(
                    f"YOLO model file not found locally: {self.model_name}. "
                    "Set allow_download=True to permit Ultralytics to download weights."
                )
            try:
                model = yolo_class(str(model_path) if model_path.is_file() else self.model_name)
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
        normalized = re.sub(r"[^a-z0-9\s_-]+", "", object_name.strip().lower())
        normalized = re.sub(r"[_-]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return YoloObjectVerifier._lemmatize_label(normalized)


    @staticmethod
    def _lemmatize_label(label: str) -> str:
        tokens = [token for token in label.split() if token]
        if not tokens:
            return ""

        lemmatized_tokens = [YoloObjectVerifier._lemmatize_token(token) for token in tokens]
        return " ".join(token for token in lemmatized_tokens if token).strip()

    @staticmethod
    def _lemmatize_token(token: str) -> str:
        if _LEMMATIZER is None:
            return YoloObjectVerifier._fallback_singularize(token)

        try:
            return _LEMMATIZER.lemmatize(token, pos="n")
        except LookupError:
            LOGGER.warning("WordNet data is unavailable; skipping lemmatization for token=%s", token)
            return YoloObjectVerifier._fallback_singularize(token)

    @staticmethod
    def _fallback_singularize(token: str) -> str:
        if len(token) <= 3:
            return token

        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"

        if token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 4:
            return token[:-2]

        if token.endswith("s") and not token.endswith(("ss", "us", "is")):
            return token[:-1]

        return token
    @classmethod
    def _is_matching_label(cls, object_name: str, detected_name: str) -> bool:
        object_variants = cls._label_variants(object_name)
        detected_variants = cls._label_variants(detected_name)
        return not object_variants.isdisjoint(detected_variants)

    @staticmethod
    def _label_variants(name: str) -> set:
        normalized = YoloObjectVerifier._normalize_object_name(name)
        if not normalized:
            return set()

        variants = {normalized}
        if normalized.endswith("es") and len(normalized) > 2:
            variants.add(normalized[:-2])
        if normalized.endswith("s") and len(normalized) > 1:
            variants.add(normalized[:-1])
        if not normalized.endswith("s"):
            variants.add(normalized + "s")
        if not normalized.endswith("es"):
            variants.add(normalized + "es")
        return {variant for variant in variants if variant}

    @staticmethod
    def _empty_result(error: Optional[str] = None) -> Dict[str, Any]:
        return {
            "verified": False,
            "confidence": 0.0,
            "matched_label": None,
            "error": error,
        }

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


def verify_object(image_path: str, object_name: str) -> Dict[str, Any]:
    """Module-level helper for one-off object verification."""
    return _DEFAULT_VERIFIER.verify_object(image_path=image_path, object_name=object_name)