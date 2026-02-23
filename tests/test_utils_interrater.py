"""Tests for compute_interrater_reliability in mathipy.utils."""

import pytest

from mathipy.utils import compute_interrater_reliability


class TestAgreement:
    def test_perfect_agreement(self):
        r = compute_interrater_reliability([1, 1, 0, 1], [1, 1, 0, 1])
        assert r["agreement"] == 1.0
        assert r["kappa"] == 1.0
        assert r["n"] == 4

    def test_no_agreement(self):
        r = compute_interrater_reliability([1, 1, 1, 1], [0, 0, 0, 0])
        assert r["agreement"] == 0.0
        # p_e = 0 when categories don't overlap, so kappa = 0
        assert r["kappa"] == 0.0

    def test_partial_agreement(self):
        r = compute_interrater_reliability([1, 1, 0, 1], [1, 0, 0, 1])
        assert 0 < r["agreement"] < 1
        assert 0 < r["kappa"] < 1
        assert r["n"] == 4


class TestKappa:
    def test_kappa_matches_manual(self):
        # 2x2: coder1=[1,1,0,0], coder2=[1,0,1,0]
        # agreement = 2/4 = 0.5
        # p_e = (2/4)*(2/4) + (2/4)*(2/4) = 0.5
        # kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0
        r = compute_interrater_reliability([1, 1, 0, 0], [1, 0, 1, 0])
        assert r["kappa"] == 0.0

    def test_multiclass(self):
        r = compute_interrater_reliability([0, 1, 2, 1, 0], [0, 1, 2, 2, 0])
        assert r["agreement"] == 0.8
        assert r["kappa"] > 0


class TestEdgeCases:
    def test_empty_sequences(self):
        r = compute_interrater_reliability([], [])
        assert r["n"] == 0
        assert r["agreement"] == 0.0
        assert r["kappa"] == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            compute_interrater_reliability([1, 0], [1])

    def test_single_element(self):
        r = compute_interrater_reliability([1], [1])
        assert r["agreement"] == 1.0
        assert r["n"] == 1

    def test_string_labels(self):
        r = compute_interrater_reliability(["a", "b", "a"], ["a", "b", "b"])
        assert r["agreement"] == pytest.approx(2 / 3, abs=0.001)
