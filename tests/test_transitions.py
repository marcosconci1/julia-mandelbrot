import pandas as pd

from juliams.analysis.transitions import analyze_regime_persistence, compute_transition_matrix


def test_empty_transition_matrix_has_defined_persistence_summary():
    df = pd.DataFrame({"regime": ["Unknown", None]})

    transition_matrix = compute_transition_matrix(df)
    persistence = analyze_regime_persistence(transition_matrix)

    assert transition_matrix.empty
    assert persistence["overall"]["max_persistence_regime"] is None
    assert persistence["overall"]["min_persistence_regime"] is None
