import pickle

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.predict import (
    ACCESS_DENIED,
    ACCESS_GRANTED,
    IMAGE_SIZE,
    authenticate,
    prepare_features,
    predict,
)


def test_prepare_features_resizes_normalizes_and_flattens():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)

    features = prepare_features(
        image=image,
        detector_fn=lambda frame: frame[2:18, 2:18],
        feature_extractor_fn=_raise_not_implemented,
    )

    assert features.shape == (1, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    assert float(features.min()) == pytest.approx(1.0)
    assert float(features.max()) == pytest.approx(1.0)


def test_predict_grants_access_for_confident_match(tmp_path):
    model_path = _build_knn_model(tmp_path)
    query_image = np.full(IMAGE_SIZE, 0.12, dtype=np.float32)

    result = authenticate(
        image=query_image,
        model_path=model_path,
        threshold=0.6,
        detector_fn=lambda frame: frame,
        feature_extractor_fn=lambda face: np.asarray(face, dtype=np.float32).reshape(-1),
    )

    assert result.predicted_label == "alice"
    assert result.label == "alice"
    assert result.access_granted is True
    assert result.decision == ACCESS_GRANTED
    assert result.confidence == pytest.approx(1.0)


def test_predict_denies_low_confidence_match(tmp_path):
    model_path = _build_knn_model(tmp_path)
    model = pickle.loads(model_path.read_bytes())
    ambiguous_image = np.full(IMAGE_SIZE, 0.45, dtype=np.float32)

    result = predict(
        model=model,
        image=ambiguous_image,
        threshold=0.8,
        detector_fn=lambda frame: frame,
        feature_extractor_fn=lambda face: np.asarray(face, dtype=np.float32).reshape(-1),
    )

    assert result.predicted_label == "alice"
    assert result.label == "Unknown"
    assert result.access_granted is False
    assert result.decision == ACCESS_DENIED
    assert result.reason == "confidence_below_threshold"
    assert result.confidence == pytest.approx(2.0 / 3.0, rel=1e-3)


def test_predict_denies_when_no_face_is_detected(tmp_path):
    model = pickle.loads(_build_knn_model(tmp_path).read_bytes())
    image = np.zeros(IMAGE_SIZE, dtype=np.float32)

    result = predict(
        model=model,
        image=image,
        detector_fn=lambda _: None,
        feature_extractor_fn=lambda face: np.asarray(face, dtype=np.float32).reshape(-1),
    )

    assert result.predicted_label is None
    assert result.label == "Unknown"
    assert result.access_granted is False
    assert result.decision == ACCESS_DENIED
    assert result.reason == "No face detected in image."


def _build_knn_model(tmp_path):
    alice_samples = [
        np.full(IMAGE_SIZE, 0.00, dtype=np.float32),
        np.full(IMAGE_SIZE, 0.10, dtype=np.float32),
        np.full(IMAGE_SIZE, 0.20, dtype=np.float32),
    ]
    bob_samples = [
        np.full(IMAGE_SIZE, 0.80, dtype=np.float32),
        np.full(IMAGE_SIZE, 0.90, dtype=np.float32),
        np.full(IMAGE_SIZE, 1.00, dtype=np.float32),
    ]

    X = np.vstack([sample.reshape(-1) for sample in alice_samples + bob_samples])
    y = np.array(["alice", "alice", "alice", "bob", "bob", "bob"])

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    model_path = tmp_path / "face_model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    return model_path


def _raise_not_implemented(_):
    raise NotImplementedError
