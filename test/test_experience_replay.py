import tempfile
from pathlib import Path
import numpy as np

from autocut import (
    load_teach_model,
    save_teach_model,
    learn_embedding_model,
    calculate_deep_scores,
    TeachModelResult,
    N_EMBEDDING_FEATURES,
)


def test_legacy_model_loading_and_backward_compatibility():
    """Verify loading legacy npz without buffer_pos/buffer_neg unpacks as 6-tuple without error."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "legacy.npz"
        w = np.random.randn(N_EMBEDDING_FEATURES)
        b = 0.5
        mean = np.random.randn(N_EMBEDDING_FEATURES)
        std = np.abs(np.random.randn(N_EMBEDDING_FEATURES)) + 0.1
        proto = np.random.randn(N_EMBEDDING_FEATURES)
        proto /= np.linalg.norm(proto)
        proto_neg = np.random.randn(N_EMBEDDING_FEATURES)
        proto_neg /= np.linalg.norm(proto_neg)
        np.savez(p, w=w, b=b, mean=mean, std=std, proto=proto, proto_neg=proto_neg)

        # 6-tuple unpack
        w_l, b_l, mean_l, std_l, proto_pos_l, proto_neg_l = load_teach_model(p)
        assert len(w_l) == N_EMBEDDING_FEATURES
        assert isinstance(b_l, float)
        assert np.allclose(mean_l, mean)

        # Check attributes
        res = load_teach_model(p)
        assert res.buffer_pos is None
        assert res.buffer_neg is None
        assert res.is_unit_norm is False


def test_modern_model_saving_and_loading():
    """Verify modern model with buffer_pos and buffer_neg saves and loads properly."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "modern.npz"
        w = np.random.randn(N_EMBEDDING_FEATURES).astype(np.float32)
        b = -1.2
        mean = np.zeros(N_EMBEDDING_FEATURES, dtype=np.float32)
        std = np.ones(N_EMBEDDING_FEATURES, dtype=np.float32)
        proto_pos = np.random.randn(N_EMBEDDING_FEATURES).astype(np.float32)
        proto_pos /= np.linalg.norm(proto_pos)
        proto_neg = np.random.randn(N_EMBEDDING_FEATURES).astype(np.float32)
        proto_neg /= np.linalg.norm(proto_neg)
        buf_pos = np.random.randn(10, N_EMBEDDING_FEATURES).astype(np.float32)
        buf_neg = np.random.randn(5, N_EMBEDDING_FEATURES).astype(np.float32)

        save_teach_model(p, w, b, mean, std, proto_pos, proto_neg, buf_pos, buf_neg)

        res = load_teach_model(p)
        assert res.is_unit_norm is True
        assert res.buffer_pos is not None
        assert len(res.buffer_pos) == 10
        assert res.buffer_neg is not None
        assert len(res.buffer_neg) == 5

        # Unpack as 6 items
        w_m, b_m, mean_m, std_m, proto_m, proto_neg_m = res
        assert len(w_m) == N_EMBEDDING_FEATURES


def test_experience_replay_accumulation():
    """Verify that training on video 2 accumulates exemplars from video 1."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "continual.npz"

        # Session 1: 5 positive frames, 2 negative
        rng = np.random.default_rng(42)
        emb1 = rng.standard_normal((100, N_EMBEDDING_FEATURES))
        centers1 = np.arange(100) * 0.48 + 0.48
        pos_labels1 = [(1.0, 2.0)]
        neg_labels1 = [(5.0, 6.0)]

        w1, b1, m1, s1, p_pos1, p_neg1, buf_pos1, buf_neg1 = learn_embedding_model(
            emb1, centers1, pos_labels1, neg_labels1
        )
        save_teach_model(model_path, w1, b1, m1, s1, p_pos1, p_neg1, buf_pos1, buf_neg1)
        initial_pos_count = len(buf_pos1)
        initial_neg_count = len(buf_neg1)
        assert initial_pos_count > 0

        # Session 2: warm start on video 2 with new positive frames
        emb2 = rng.standard_normal((100, N_EMBEDDING_FEATURES))
        centers2 = np.arange(100) * 0.48 + 0.48
        pos_labels2 = [(10.0, 11.5)]
        neg_labels2 = [(15.0, 16.0)]

        w2, b2, m2, s2, p_pos2, p_neg2, buf_pos2, buf_neg2 = learn_embedding_model(
            emb2, centers2, pos_labels2, neg_labels2, warm_start_path=model_path
        )
        assert len(buf_pos2) > initial_pos_count, "Session 2 must accumulate previous positive exemplars"
        assert len(buf_neg2) > initial_neg_count, "Session 2 must accumulate previous negative exemplars"


def test_prototype_anchor_gating_crushes_false_positives():
    """Verify that an out-of-distribution loud noise cannot produce high score even if linear dot product is high."""
    rng = np.random.default_rng(123)

    proto_pos = rng.standard_normal(N_EMBEDDING_FEATURES)
    proto_pos /= np.linalg.norm(proto_pos)

    # Let w be orthogonal to proto_pos with large magnitude
    w = rng.standard_normal(N_EMBEDDING_FEATURES)
    w = w - np.dot(w, proto_pos) * proto_pos
    w = w / np.linalg.norm(w) * 15.0
    b = 0.0
    mean = np.zeros(N_EMBEDDING_FEATURES)
    std = np.ones(N_EMBEDDING_FEATURES)

    # An unrelated loud sound aligning with w:
    loud_unrelated_sound = (w / np.linalg.norm(w)).reshape(1, -1)
    # Dot product with w is +15, so linear probability is sigmoid(15) ~ 0.999999 (100%!)
    # But dot product with proto_pos is 0.0 (cosine similarity = 0)

    score = calculate_deep_scores(
        loud_unrelated_sound, w, b, mean, std, proto_pos=proto_pos, is_unit_norm=True
    )
    assert score[0] < 0.01, f"Unrelated sound must be crushed by prototype gate, got {score[0]}"


def test_contrastive_negative_suppression():
    """Verify that a sound closer to negative prototype is suppressed even if linear wants to trigger."""
    proto_pos = np.zeros(N_EMBEDDING_FEATURES)
    proto_pos[0] = 1.0

    proto_neg = np.zeros(N_EMBEDDING_FEATURES)
    proto_neg[1] = 1.0

    # Sound closer to negative prototype (sim_pos = 0.6, sim_neg = 0.8)
    sound = np.zeros((1, N_EMBEDDING_FEATURES))
    sound[0, 0] = 0.6
    sound[0, 1] = 0.8
    sound = sound / np.linalg.norm(sound)

    w = np.zeros(N_EMBEDDING_FEATURES)
    b = 5.0 # Linear wants to say YES (sigmoid(5) ~ 0.993)
    mean = np.zeros(N_EMBEDDING_FEATURES)
    std = np.ones(N_EMBEDDING_FEATURES)

    score = calculate_deep_scores(
        sound, w, b, mean, std, proto_pos=proto_pos, proto_neg=proto_neg, is_unit_norm=True
    )
    assert score[0] < 0.05, f"Sound closer to negative prototype must be suppressed, got {score[0]}"


if __name__ == "__main__":
    test_legacy_model_loading_and_backward_compatibility()
    test_modern_model_saving_and_loading()
    test_experience_replay_accumulation()
    test_prototype_anchor_gating_crushes_false_positives()
    test_contrastive_negative_suppression()
    print("All unit tests passed successfully!")
