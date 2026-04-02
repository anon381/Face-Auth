from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from config import MODEL_NAME, MODELS_DIR
from src.feature_engineering import extract_features
from src.preprocessing import detect_and_align

IMAGE_SIZE = (64, 64)
DEFAULT_THRESHOLD = 0.6
UNKNOWN_LABEL = "Unknown"
ACCESS_GRANTED = "Access Granted"
ACCESS_DENIED = "Access Denied"


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str | None
    label: str
    confidence: float
    threshold: float
    access_granted: bool
    decision: str
    reason: str


def load_model(model_path: str | Path | None = None) -> Any:
    """Load a persisted classifier and validate its inference interface."""
    resolved_path = Path(model_path) if model_path is not None else MODELS_DIR / MODEL_NAME

    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    with resolved_path.open("rb") as model_file:
        model = pickle.load(model_file)

    if not hasattr(model, "predict"):
        raise TypeError("Loaded object must implement a 'predict' method.")

    return model


def prepare_features(
    image: Any,
    detector_fn: Callable[[Any], Any] = detect_and_align,
    feature_extractor_fn: Callable[[Any], Any] = extract_features,
) -> np.ndarray:
    """Convert an input image into the feature vector expected by the model."""
    raw_image = _ensure_numpy_image(image)
    face_image = _detect_face(raw_image, detector_fn)
    return _extract_model_features(face_image, feature_extractor_fn)


def predict(
    model: Any,
    image: Any,
    threshold: float = DEFAULT_THRESHOLD,
    detector_fn: Callable[[Any], Any] = detect_and_align,
    feature_extractor_fn: Callable[[Any], Any] = extract_features,
    unknown_label: str = UNKNOWN_LABEL,
) -> PredictionResult:
    """Predict identity and apply threshold-based access control."""
    _validate_threshold(threshold)

    try:
        features = prepare_features(
            image=image,
            detector_fn=detector_fn,
            feature_extractor_fn=feature_extractor_fn,
        )
    except ValueError as exc:
        return PredictionResult(
            predicted_label=None,
            label=unknown_label,
            confidence=0.0,
            threshold=threshold,
            access_granted=False,
            decision=ACCESS_DENIED,
            reason=str(exc),
        )

    predicted_values = np.asarray(model.predict(features)).ravel()
    if predicted_values.size == 0:
        raise ValueError("Model returned an empty prediction.")

    predicted_label = str(predicted_values[0])
    confidence = _estimate_confidence(model, features, predicted_label)
    access_granted = confidence >= threshold

    return PredictionResult(
        predicted_label=predicted_label,
        label=predicted_label if access_granted else unknown_label,
        confidence=confidence,
        threshold=threshold,
        access_granted=access_granted,
        decision=ACCESS_GRANTED if access_granted else ACCESS_DENIED,
        reason="match" if access_granted else "confidence_below_threshold",
    )


def authenticate(
    image: Any,
    model_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    detector_fn: Callable[[Any], Any] = detect_and_align,
    feature_extractor_fn: Callable[[Any], Any] = extract_features,
    unknown_label: str = UNKNOWN_LABEL,
) -> PredictionResult:
    """Load the saved model and run the full inference pipeline."""
    model = load_model(model_path)
    return predict(
        model=model,
        image=image,
        threshold=threshold,
        detector_fn=detector_fn,
        feature_extractor_fn=feature_extractor_fn,
        unknown_label=unknown_label,
    )


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")


def _ensure_numpy_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Input image is empty.")
    return array


def _detect_face(image: np.ndarray, detector_fn: Callable[[Any], Any]) -> np.ndarray:
    try:
        detected = detector_fn(image)
    except NotImplementedError:
        detected = image

    if detected is None:
        raise ValueError("No face detected in image.")

    detected_array = np.asarray(detected)
    if detected_array.size == 0:
        raise ValueError("Detected face is empty.")

    return detected_array


def _extract_model_features(
    face_image: np.ndarray,
    feature_extractor_fn: Callable[[Any], Any],
) -> np.ndarray:
    try:
        extracted = feature_extractor_fn(face_image)
    except NotImplementedError:
        extracted = _fallback_extract_features(face_image)

    if extracted is None:
        extracted = _fallback_extract_features(face_image)

    feature_array = np.asarray(extracted, dtype=np.float32)

    if feature_array.ndim > 1:
        feature_array = feature_array.reshape(-1)

    if feature_array.size == 0:
        raise ValueError("Feature extraction produced no values.")

    if float(np.max(feature_array)) > 1.0:
        feature_array = feature_array / 255.0

    return feature_array.reshape(1, -1)


def _fallback_extract_features(face_image: np.ndarray) -> np.ndarray:
    image = np.asarray(face_image)

    if image.ndim == 1:
        side = int(np.sqrt(image.size))
        if side * side != image.size:
            raise ValueError("Flattened input cannot be reshaped into a square face image.")
        image = image.reshape(side, side)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError("Face image must be 2D grayscale or 3D color data.")

    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32)

    if float(np.max(normalized)) > 1.0:
        normalized = normalized / 255.0

    return normalized.reshape(-1)


def _estimate_confidence(model: Any, features: np.ndarray, predicted_label: str) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float32)
        if probabilities.ndim == 2 and probabilities.shape[0] > 0:
            class_labels = getattr(model, "classes_", None)
            if class_labels is not None:
                for index, class_label in enumerate(class_labels):
                    if str(class_label) == predicted_label:
                        return float(probabilities[0][index])
            return float(np.max(probabilities[0]))

    if hasattr(model, "kneighbors"):
        distances, _ = model.kneighbors(features)
        mean_distance = float(np.mean(distances[0]))
        return float(1.0 / (1.0 + max(mean_distance, 0.0)))

    return 1.0
