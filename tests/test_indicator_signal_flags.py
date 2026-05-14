import numpy as np
import pandas as pd

from juliams.features.trend import compute_trend_features
from juliams.features.volatility import compute_volatility_features
from juliams.regimes.classification import RegimeClassifier
from juliams.regimes.quality import add_indicator_signal_flags


def test_indicator_flags_mark_warmup_and_ready_rows():
    df = pd.DataFrame(
        {
            "trend_strength": [np.nan, 0.4, 0.5, 0.6, 0.7],
            "volatility": [np.nan, 0.1, 0.1, 0.1, 0.1],
            "volatility_percentile": [np.nan, 0.2, 0.8, 0.8, 0.8],
            "regime": ["Unknown", "Up-LowVol", "Up-HighVol", "Up-HighVol", "Up-HighVol"],
        }
    )

    out = add_indicator_signal_flags(
        df,
        {
            "trend_window": 2,
            "volatility_window": 2,
            "volatility_percentile_lookback": 4,
        },
    )

    assert out["warmup_bars"].iloc[-1] == 4
    assert out["warmup_complete"].tolist() == [False, False, False, True, True]
    assert out["valid_signal"].tolist() == [False, False, False, True, True]
    assert out["calibration_observations"].tolist() == [1, 2, 3, 4, 5]
    assert out["causal_mode"].all()


def test_core_indicator_pipeline_does_not_rewrite_past_after_future_changes():
    cfg = {
        "trend_window": 10,
        "volatility_window": 10,
        "volatility_percentile_lookback": 20,
        "volatility_baseline_window": 20,
    }
    base = _price_frame()
    poisoned = base.copy()
    poisoned.iloc[70:, poisoned.columns.get_loc("Close")] *= 1.8

    clean_out = _classify(base, cfg)
    poisoned_out = _classify(poisoned, cfg)

    columns = ["regime", "warmup_complete", "valid_signal"]
    pd.testing.assert_frame_equal(clean_out.loc[:base.index[59], columns], poisoned_out.loc[:base.index[59], columns])


def _classify(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = compute_trend_features(df, cfg)
    out = compute_volatility_features(out, cfg)
    out = RegimeClassifier().classify(out)
    return add_indicator_signal_flags(out, cfg)


def _price_frame() -> pd.DataFrame:
    close = 100.0 + np.cumsum(np.sin(np.arange(100) / 5.0) * 0.2 + 0.05)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="h"),
    )
