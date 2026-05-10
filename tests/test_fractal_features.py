import numpy as np
import pandas as pd

from juliams.features.fractal import create_fractal_mask, compute_fractal_filtered_price


def test_fractal_mask_smoothing_uses_trailing_window_only():
    hurst = pd.Series([0.0, 1.0, 1.0])

    mask = create_fractal_mask(hurst, threshold=0.55, smooth=True, smooth_window=3)

    assert mask.tolist() == [0.0, 1.0, 1.0]


def test_fractal_filtered_price_starts_at_initial_price():
    df = pd.DataFrame(
        {
            "Close": [100.0, 105.0, 110.0],
            "hurst": [0.8, 0.8, 0.8],
        }
    )

    filtered = compute_fractal_filtered_price(df, threshold=0.55, method="accumulate")

    assert np.allclose(filtered.to_numpy(), df["Close"].to_numpy())
