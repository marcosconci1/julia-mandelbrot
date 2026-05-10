"""Tests for juliams.regimes.bocpd_calibration."""

from __future__ import annotations

import pytest

from juliams.regimes.bocpd_calibration import (
    DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET,
    default_bocpd_expected_run_length,
)


def test_defaults_present_for_main_asset_classes():
    for asset in ("equity", "fx", "commodity", "crypto"):
        assert default_bocpd_expected_run_length(asset) > 0


def test_defaults_ordering_matches_documented_volatility_ranking():
    """Crypto < commodity < FX < equity in expected run length, since
    higher-vol assets need a more reactive (smaller) prior."""
    eq = default_bocpd_expected_run_length("equity")
    fx = default_bocpd_expected_run_length("fx")
    co = default_bocpd_expected_run_length("commodity")
    cr = default_bocpd_expected_run_length("crypto")
    assert cr < co < fx < eq


def test_unknown_asset_raises():
    with pytest.raises(ValueError, match="Unknown asset_class"):
        default_bocpd_expected_run_length("nfts")


def test_case_insensitive():
    assert default_bocpd_expected_run_length("EQUITY") == default_bocpd_expected_run_length("equity")
    assert default_bocpd_expected_run_length(" Crypto ") == default_bocpd_expected_run_length("crypto")


def test_dict_is_dict():
    """Sanity-check the public dict is what we expect."""
    assert set(DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET) == {"equity", "fx", "commodity", "crypto"}
    assert all(isinstance(v, float) for v in DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET.values())
