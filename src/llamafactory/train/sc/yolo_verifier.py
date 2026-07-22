import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import embedding

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
        semantic_similarity_threshold: float = 0.72,
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.allow_download = allow_download
        self._model = None
        self._semantic_model = None
        self._semantic_model_failed = False
        self._device: Optional[str] = None
        self._cache: Dict[Path, List[Dict[str, Any]]] = {}
        self._embedding_cache: Dict[str, Tuple[float, ...]] = {}
        self._lock = threading.Lock()
        self._semantic_lock = threading.Lock()

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
        best_similarity = 0.0
        match_type = "none"
        for detection in detections:
            detected_name = self._normalize_object_name(detection["name"])
            if self._is_matching_label(normalized_object, detected_name):
                best_confidence = max(best_confidence, detection["confidence"])
                if detection["confidence"] == best_confidence:
                    matched_label = detected_name
                    best_similarity = 1.0
                    match_type = "exact"

        if matched_label is None:
            for detection in detections:
                detected_name = self._normalize_object_name(detection["name"])
                similarity = self._semantic_similarity(normalized_object, detected_name)
                if similarity > 0.5:
                    LOGGER.debug(
                        "Semantic similarity: object=%s detected=%s similarity=%.4f",
                        normalized_object,
                        detected_name,
                        similarity,
                    )

                if similarity > best_similarity:
                    best_similarity = similarity

                if similarity >= self.semantic_similarity_threshold:
                    detection_confidence = float(detection["confidence"])
                    LOGGER.info(
                        "Semantic verification match image=%s object=%s detected=%s "
                        "similarity=%.4f confidence=%.4f threshold=%.2f",
                        image,
                        normalized_object,
                        detected_name,
                        similarity,
                        detection_confidence,
                        self.semantic_similarity_threshold,
                    )
                    if detection_confidence >= best_confidence:
                        best_confidence = detection_confidence
                        matched_label = detected_name
                        match_type = "semantic"

        verified = best_confidence >= self.confidence_threshold
        if match_type == "semantic":
            LOGGER.info(
                "YOLO semantic verification SUCCESS image=%s object=%s "
                "matched_label=%s similarity=%.4f confidence=%.4f",
                image,
                normalized_object,
                matched_label,
                best_similarity,
                best_confidence,
            )

        result = {
            "verified": verified, 
            "confidence": float(best_confidence),
            "matched_label": matched_label,
            "semantic_similarity": float(best_similarity) if matched_label is not None else best_similarity,
            "match_type": match_type,
            "error": None,
        }
        LOGGER.debug(
            "YOLO verification result image=%s object=%s verified=%s confidence=%.4f matched_label=%s match_type=%s semantic_similarity=%.4f",
            image,
            normalized_object,
            verified,
            best_confidence,
            matched_label,
            match_type,
            result["semantic_similarity"]
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

    def _get_semantic_model(self) -> Optional[Any]:
        with self._semantic_lock:
            if self._semantic_model is not None:
                return self._semantic_model

            if self._semantic_model_failed:
                return None

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                LOGGER.warning(
                    "sentence-transformers is not available; semantic fallback matching is disabled"
                )
                self._semantic_model_failed = True
                return None

            try:
                LOGGER.info(
                    "Loading semantic similarity model=%s",
                    "sentence-transformers/all-mpnet-base-v2",
                )
                self._semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            except Exception as exc:
                LOGGER.warning("Failed to load semantic similarity model: %s", exc)
                self._semantic_model_failed = True
                return None

            return self._semantic_model

    def _semantic_similarity(self, object_name: str, detected_label: str) -> float:
        normalized_object = self._normalize_object_name(object_name)
        normalized_detected = self._normalize_object_name(detected_label)
        if not normalized_object or not normalized_detected:
            return 0.0

        if normalized_object == normalized_detected:
            return 1.0

        object_embedding = self._get_text_embedding(normalized_object)
        detected_embedding = self._get_text_embedding(normalized_detected)
        if object_embedding is None or detected_embedding is None:
            return 0.0

        return max(0.0, min(1.0, sum(left * right for left, right in zip(object_embedding, detected_embedding))))

    def _get_text_embedding(self, label: str) -> Optional[Tuple[float, ...]]:
        normalized_label = self._normalize_object_name(label)
        if not normalized_label:
            return None

        with self._semantic_lock:
            cached_embedding = self._embedding_cache.get(normalized_label)
            if cached_embedding is not None:
                return cached_embedding

        model = self._get_semantic_model()
        if model is None:
            return None

        try:
            embedding = model.encode(normalized_label, normalize_embeddings=True)
        except Exception as exc:
            LOGGER.warning("Failed to encode semantic label=%s: %s", normalized_label, exc)
            return None

        normalized_embedding = self._to_embedding_tuple(embedding)
        with self._semantic_lock:
            self._embedding_cache[normalized_label] = normalized_embedding
        return normalized_embedding

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
            "semantic_similarity": 0.0,
            "match_type": "none",
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

    @staticmethod
    def _to_embedding_tuple(values: Any) -> Tuple[float, ...]:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()

        if values and isinstance(values[0], list):
            values = values[0]

        return tuple(float(value) for value in values )

_DEFAULT_VERIFIER = YoloObjectVerifier()


def verify_object(image_path: str, object_name: str) -> Dict[str, Any]:
    """Module-level helper for one-off object verification."""
    return _DEFAULT_VERIFIER.verify_object(image_path=image_path, object_name=object_name)