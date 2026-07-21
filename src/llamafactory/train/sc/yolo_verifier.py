import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import inflect
except ImportError:
    inflect = None

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

LOGGER = logging.getLogger(__name__)

_INFLECT_ENGINE = inflect.engine() if inflect is not None else None
_FUZZY_MATCH_THRESHOLD = 93.0

_PHRASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "background",
    "center",
    "edge",
    "foreground",
    "image",
    "left",
    "middle",
    "of",
    "part",
    "right",
    "side",
    "the",
    "top",
}

# Semantic detector mappings still need a curated layer because generic language
# libraries do not know that, for example, "seagull" should map to COCO "bird".
_TARGET_LABEL_ALIASES = {
    "airplane": {"airplane", "airplanes", "aeroplane", "aeroplanes", "plane", "planes", "jet", "jets"},
    "apple": {"apple", "apples"},
    "backpack": {"backpack", "backpacks", "bag", "bags", "rucksack", "rucksacks"},
    "banana": {"banana", "bananas"},
    "baseball bat": {"baseball bat", "baseball bats", "bat", "bats"},
    "baseball glove": {"baseball glove", "baseball gloves", "glove", "gloves"},
    "bear": {"bear", "bears"},
    "bed": {"bed", "beds"},
    "bench": {"bench", "benches"},
    "bicycle": {"bicycle", "bicycles", "bike", "bikes", "cycle", "cycles"},
    "bird": {"bird", "birds", "duck", "ducks", "gull", "gulls", "seagull", "seagulls"},
    "boat": {"boat", "boats", "ship", "ships", "vessel", "vessels"},
    "book": {"book", "books"},
    "bottle": {"bottle", "bottles"},
    "bowl": {"bowl", "bowls"},
    "broccoli": {"broccoli"},
    "bus": {"bus", "buses"},
    "cake": {"cake", "cakes"},
    "car": {"car", "cars", "automobile", "automobiles", "sedan", "sedans"},
    "carrot": {"carrot", "carrots"},
    "cat": {"cat", "cats", "kitten", "kittens"},
    "cell phone": {"cell phone", "cell phones", "mobile phone", "mobile phones", "phone", "phones", "smartphone", "smartphones"},
    "clock": {"clock", "clocks", "watch", "watches"},
    "couch": {"couch", "couches", "sofa", "sofas"},
    "cow": {"cow", "cows"},
    "cup": {"cup", "cups", "mug", "mugs"},
    "dining table": {"dining table", "dining tables", "table", "tables", "ping pong table", "ping pong tables", "table tennis table", "table tennis tables"},
    "dog": {"dog", "dogs", "puppy", "puppies"},
    "donut": {"donut", "donuts", "doughnut", "doughnuts"},
    "elephant": {"elephant", "elephants"},
    "fire hydrant": {"fire hydrant", "fire hydrants", "hydrant", "hydrants"},
    "fork": {"fork", "forks"},
    "frisbee": {"frisbee", "frisbees"},
    "giraffe": {"giraffe", "giraffes"},
    "hair drier": {"hair drier", "hair driers", "hair dryer", "hair dryers", "dryer", "dryers"},
    "handbag": {"handbag", "handbags", "purse", "purses", "bag", "bags"},
    "horse": {"horse", "horses"},
    "hot dog": {"hot dog", "hot dogs"},
    "keyboard": {"keyboard", "keyboards"},
    "kite": {"kite", "kites"},
    "knife": {"knife", "knives"},
    "laptop": {"laptop", "laptops", "notebook", "notebooks", "computer", "computers"},
    "microwave": {"microwave", "microwaves"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "motorbikes"},
    "mouse": {"mouse", "mice", "computer mouse", "computer mice"},
    "orange": {"orange", "oranges"},
    "oven": {"oven", "ovens"},
    "parking meter": {"parking meter", "parking meters", "meter", "meters"},
    "person": {"person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children", "pedestrian", "pedestrians"},
    "pizza": {"pizza", "pizzas"},
    "potted plant": {"potted plant", "potted plants", "plant", "plants", "tree", "trees", "bush", "bushes"},
    "refrigerator": {"refrigerator", "refrigerators", "fridge", "fridges"},
    "remote": {"remote", "remotes", "remote control", "remote controls"},
    "sandwich": {"sandwich", "sandwiches"},
    "scissors": {"scissors"},
    "sheep": {"sheep", "ram", "rams"},
    "sink": {"sink", "sinks"},
    "skateboard": {"skateboard", "skateboards"},
    "skis": {"ski", "skis"},
    "snowboard": {"snowboard", "snowboards"},
    "spoon": {"spoon", "spoons"},
    "sports ball": {"sports ball", "sports balls", "ball", "balls"},
    "stop sign": {"stop sign", "stop signs"},
    "suitcase": {"suitcase", "suitcases", "luggage", "baggage"},
    "surfboard": {"surfboard", "surfboards", "board", "boards"},
    "teddy bear": {"teddy bear", "teddy bears", "bear", "bears", "stuffed bear", "stuffed bears"},
    "tennis racket": {"tennis racket", "tennis rackets", "racket", "rackets"},
    "tie": {"tie", "ties", "necktie", "neckties"},
    "toaster": {"toaster", "toasters"},
    "toilet": {"toilet", "toilets"},
    "toothbrush": {"toothbrush", "toothbrushes"},
    "traffic light": {"traffic light", "traffic lights", "signal light", "signal lights", "street light", "street lights", "light", "lights"},
    "train": {"train", "trains", "locomotive", "locomotives"},
    "truck": {"truck", "trucks", "lorry", "lorries"},
    "tv": {"tv", "tvs", "television", "televisions", "tv screen", "tv screens", "television screen", "television screens", "monitor", "monitors", "screen", "screens"},
    "vase": {"vase", "vases"},
    "wine glass": {"wine glass", "wine glasses", "glass", "glasses"},
    "zebra": {"zebra", "zebras"},
}

_MULTI_LABEL_ALIASES = {
    "vehicle": {"car", "bus", "truck", "motorcycle", "bicycle", "train"},
    "vehicles": {"car", "bus", "truck", "motorcycle", "bicycle", "train"},
}


class YoloVerificationError(RuntimeError):
    """Raised when image verification cannot be completed."""


class YoloObjectVerifier:
    """Lazy YOLOv8n verifier with per-image detection caching."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
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
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _is_matching_label(cls, object_name: str, detected_name: str) -> bool:
        for object_candidate in cls._candidate_labels(object_name):
            for detected_candidate in cls._candidate_labels(detected_name):
                if object_candidate == detected_candidate:
                    return True

                object_variants = cls._label_variants(object_candidate)
                detected_variants = cls._label_variants(detected_candidate)
                if not object_variants.isdisjoint(detected_variants):
                    return True

            if cls._is_fuzzy_match(object_candidate, detected_candidate):
                return True

        return False
    
    @classmethod
    def _candidate_labels(cls, object_name: str) -> set:
        normalized = cls._normalize_object_name(object_name)
        if not normalized:
            return set()

        candidates = {normalized}
        candidates.update(_MULTI_LABEL_ALIASES.get(normalized, set()))

        phrase_candidates = cls._phrase_candidates(normalized)
        for phrase in phrase_candidates:
            candidates.add(phrase)
            candidates.update(_MULTI_LABEL_ALIASES.get(phrase, set()))

        normalized_variants = set()
        for phrase in phrase_candidates | {normalized}:
            normalized_variants.update(cls._label_variants(phrase))

        for target_label, aliases in _TARGET_LABEL_ALIASES.items():
            alias_variants = {cls._normalize_object_name(target_label)}
            for alias in aliases:
                alias_variants.update(cls._label_variants(alias))

            if not normalized_variants.isdisjoint(alias_variants):
                candidates.add(target_label)

        return {candidate for candidate in candidates if candidate}

    @classmethod
    def _phrase_candidates(cls, normalized: str) -> set:
        tokens = [token for token in normalized.split() if token]
        if not tokens:
            return set()

        phrases = set(tokens)
        informative_tokens = [token for token in tokens if token not in _PHRASE_STOPWORDS]
        phrases.update(informative_tokens)
        if informative_tokens:
            phrases.add(informative_tokens[-1])

        window_source = informative_tokens or tokens
        max_window = min(3, len(window_source))
        for window_size in range(2, max_window + 1):
            for start in range(len(window_source) - window_size + 1):
                phrases.add(" ".join(window_source[start : start + window_size]))

        return {phrase for phrase in phrases if phrase}

    @staticmethod
    def _label_variants(name: str) -> set:
        normalized = YoloObjectVerifier._normalize_object_name(name)
        if not normalized:
            return set()

        variants = {normalized}
        if _INFLECT_ENGINE is not None:
            singular = _INFLECT_ENGINE.singular_noun(normalized)
            plural = _INFLECT_ENGINE.plural(normalized)
            if isinstance(singular, str) and singular:
                variants.add(YoloObjectVerifier._normalize_object_name(singular))
            if isinstance(plural, str) and plural:
                variants.add(YoloObjectVerifier._normalize_object_name(plural))
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
    def _is_fuzzy_match(left: str, right: str) -> bool:
        if fuzz is None:
            return False

        if not left or not right:
            return False

        score = max(
            float(fuzz.ratio(left, right)),
            float(fuzz.token_sort_ratio(left, right)),
        )
        return score >= _FUZZY_MATCH_THRESHOLD

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