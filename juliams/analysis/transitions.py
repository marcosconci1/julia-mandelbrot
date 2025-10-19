"""
Regime transition analysis for the Julia Mandelbrot System.
Computes Markov transition matrices and analyzes regime dynamics.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_transition_matrix(df: pd.DataFrame,
                             regime_col: str = 'regime',
                             normalize: bool = True) -> pd.DataFrame:
    """
    Compute the regime transition probability matrix.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime classification
        normalize: Whether to normalize to probabilities
    
    Returns:
        DataFrame with transition matrix (rows=current, cols=next)
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    
    # Get regime sequence
    regimes = df[regime_col].dropna()
    
    # Get unique regime values
    unique_regimes = sorted(regimes.unique())
    unique_regimes = [r for r in unique_regimes if r != 'Unknown']
    
    # Initialize transition count matrix
    n_regimes = len(unique_regimes)
    transition_counts = pd.DataFrame(
        np.zeros((n_regimes, n_regimes)),
        index=unique_regimes,
        columns=unique_regimes
    )
    
    # Count transitions
    for i in range(len(regimes) - 1):
        current_regime = regimes.iloc[i]
        next_regime = regimes.iloc[i + 1]
        
        if current_regime != 'Unknown' and next_regime != 'Unknown':
            if current_regime in unique_regimes and next_regime in unique_regimes:
                transition_counts.loc[current_regime, next_regime] += 1
    
    # Normalize to probabilities if requested
    if normalize:
        row_sums = transition_counts.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_counts.div(row_sums, axis=0)
    else:
        transition_matrix = transition_counts
    
    logger.info(f"Computed {n_regimes}x{n_regimes} transition matrix")
    
    return transition_matrix


def compute_n_step_transition(transition_matrix: pd.DataFrame, 
                             n: int) -> pd.DataFrame:
    """
    Compute n-step transition probabilities.
    
    Args:
        transition_matrix: One-step transition probability matrix
        n: Number of steps
    
    Returns:
        n-step transition probability matrix
    """
    if n < 1:
        raise ValueError("Number of steps must be at least 1")
    
    if n == 1:
        return transition_matrix
    
    # Matrix power for n-step transitions
    result = transition_matrix.copy()
    for _ in range(n - 1):
        result = result.dot(transition_matrix)
    
    return result


def compute_stationary_distribution(transition_matrix: pd.DataFrame,
                                   max_iterations: int = 1000,
                                   tolerance: float = 1e-8) -> pd.Series:
    """
    Compute the stationary distribution of the Markov chain.
    
    Args:
        transition_matrix: Transition probability matrix
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
    
    Returns:
        Series with stationary distribution
    """
    n_states = len(transition_matrix)
    
    # Start with uniform distribution
    distribution = pd.Series(
        np.ones(n_states) / n_states,
        index=transition_matrix.index
    )
    
    # Power iteration method
    for i in range(max_iterations):
        new_distribution = distribution.dot(transition_matrix)
        
        # Check convergence
        if np.abs(new_distribution - distribution).max() < tolerance:
            logger.info(f"Stationary distribution converged in {i+1} iterations")
            return new_distribution
        
        distribution = new_distribution
    
    logger.warning(f"Stationary distribution did not converge in {max_iterations} iterations")
    return distribution


def analyze_regime_persistence(transition_matrix: pd.DataFrame) -> Dict:
    """
    Analyze regime persistence from transition matrix.
    
    Args:
        transition_matrix: Transition probability matrix
    
    Returns:
        Dictionary with persistence metrics
    """
    persistence = {}
    
    # Diagonal elements represent staying in same regime
    for regime in transition_matrix.index:
        persistence[regime] = {
            'self_transition_prob': transition_matrix.loc[regime, regime],
            'expected_duration': 1 / (1 - transition_matrix.loc[regime, regime]) 
                                if transition_matrix.loc[regime, regime] < 1 else np.inf
        }
    
    # Overall persistence (average of diagonal)
    persistence['overall'] = {
        'avg_self_transition': np.diag(transition_matrix).mean(),
        'max_persistence_regime': transition_matrix.index[np.diag(transition_matrix).argmax()],
        'min_persistence_regime': transition_matrix.index[np.diag(transition_matrix).argmin()]
    }
    
    return persistence


def identify_regime_cycles(transition_matrix: pd.DataFrame,
                         threshold: float = 0.1) -> List[List[str]]:
    """
    Identify common regime cycles or sequences.
    
    Args:
        transition_matrix: Transition probability matrix
        threshold: Minimum probability to consider a transition
    
    Returns:
        List of regime cycles
    """
    cycles = []
    
    # Look for 2-cycles (A -> B -> A)
    for i, regime1 in enumerate(transition_matrix.index):
        for j, regime2 in enumerate(transition_matrix.columns):
            if i != j:  # Different regimes
                prob_1to2 = transition_matrix.iloc[i, j]
                prob_2to1 = transition_matrix.iloc[j, i]
                
                if prob_1to2 > threshold and prob_2to1 > threshold:
                    cycle = sorted([regime1, regime2])
                    if cycle not in cycles:
                        cycles.append(cycle)
    
    # Look for 3-cycles (A -> B -> C -> A)
    for i, regime1 in enumerate(transition_matrix.index):
        for j, regime2 in enumerate(transition_matrix.columns):
            for k, regime3 in enumerate(transition_matrix.index):
                if i != j and j != k and k != i:  # All different
                    prob_1to2 = transition_matrix.iloc[i, j]
                    prob_2to3 = transition_matrix.iloc[j, k]
                    prob_3to1 = transition_matrix.iloc[k, i]
                    
                    if all(p > threshold for p in [prob_1to2, prob_2to3, prob_3to1]):
                        cycle = [regime1, regime2, regime3]
                        # Check if this cycle is already recorded (in any rotation)
                        is_new = True
                        for existing in cycles:
                            if len(existing) == 3 and set(existing) == set(cycle):
                                is_new = False
                                break
                        if is_new:
                            cycles.append(cycle)
    
    return cycles


def compute_mean_first_passage_time(transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean first passage times between regimes.
    
    This calculates the expected number of steps to reach regime j
    starting from regime i for the first time.
    
    Args:
        transition_matrix: Transition probability matrix
    
    Returns:
        DataFrame with mean first passage times
    """
    n = len(transition_matrix)
    P = transition_matrix.values
    I = np.eye(n)
    
    # Initialize MFPT matrix
    mfpt = pd.DataFrame(
        np.zeros((n, n)),
        index=transition_matrix.index,
        columns=transition_matrix.columns
    )
    
    # Compute MFPT for each target state
    for j in range(n):
        # Remove j-th row and column
        P_reduced = np.delete(np.delete(P, j, axis=0), j, axis=1)
        I_reduced = np.eye(n - 1)
        
        try:
            # Solve (I - P_reduced) * m = 1
            fundamental = np.linalg.inv(I_reduced - P_reduced)
            times = fundamental.sum(axis=1)
            
            # Fill in the MFPT matrix
            idx = 0
            for i in range(n):
                if i != j:
                    mfpt.iloc[i, j] = times[idx]
                    idx += 1
        except np.linalg.LinAlgError:
            logger.warning(f"Could not compute MFPT for target regime {transition_matrix.columns[j]}")
            mfpt.iloc[:, j] = np.nan
    
    return mfpt


def analyze_regime_transitions(df: pd.DataFrame,
                              regime_col: str = 'regime') -> Dict:
    """
    Comprehensive analysis of regime transitions.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime classification
    
    Returns:
        Dictionary with comprehensive transition analysis
    """
    # Compute transition matrix
    transition_matrix = compute_transition_matrix(df, regime_col)
    
    results = {
        'transition_matrix': transition_matrix.to_dict(),
        'n_transitions': (transition_matrix.values > 0).sum() - len(transition_matrix),  # Exclude diagonal
    }
    
    # Persistence analysis
    persistence = analyze_regime_persistence(transition_matrix)
    results['persistence'] = persistence
    
    # Stationary distribution
    try:
        stationary = compute_stationary_distribution(transition_matrix)
        results['stationary_distribution'] = stationary.to_dict()
    except Exception as e:
        logger.warning(f"Could not compute stationary distribution: {e}")
        results['stationary_distribution'] = None
    
    # Most likely transitions
    likely_transitions = []
    for i, current in enumerate(transition_matrix.index):
        for j, next_regime in enumerate(transition_matrix.columns):
            if i != j:  # Exclude self-transitions
                prob = transition_matrix.iloc[i, j]
                if prob > 0.1:  # Threshold for "likely"
                    likely_transitions.append({
                        'from': current,
                        'to': next_regime,
                        'probability': prob
                    })
    
    likely_transitions.sort(key=lambda x: x['probability'], reverse=True)
    results['likely_transitions'] = likely_transitions[:10]  # Top 10
    
    # Regime cycles
    cycles = identify_regime_cycles(transition_matrix)
    results['cycles'] = cycles
    
    # Mean first passage times
    try:
        mfpt = compute_mean_first_passage_time(transition_matrix)
        results['mean_first_passage_times'] = mfpt.to_dict()
    except Exception as e:
        logger.warning(f"Could not compute mean first passage times: {e}")
        results['mean_first_passage_times'] = None
    
    # Transition entropy (measure of randomness)
    def entropy(probs):
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    
    regime_entropies = {}
    for regime in transition_matrix.index:
        regime_entropies[regime] = entropy(transition_matrix.loc[regime].values)
    
    results['transition_entropy'] = {
        'by_regime': regime_entropies,
        'overall': np.mean(list(regime_entropies.values()))
    }
    
    return results


def get_transition_statistics(df: pd.DataFrame,
                             regime_col: str = 'regime') -> Dict:
    """
    Get summary statistics for regime transitions.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime classification
    
    Returns:
        Dictionary with transition statistics
    """
    stats = {}
    
    # Basic transition analysis
    transition_analysis = analyze_regime_transitions(df, regime_col)
    stats.update(transition_analysis)
    
    # Count actual transitions
    regimes = df[regime_col].dropna()
    transitions = []
    
    for i in range(len(regimes) - 1):
        if regimes.iloc[i] != regimes.iloc[i + 1]:
            if regimes.iloc[i] != 'Unknown' and regimes.iloc[i + 1] != 'Unknown':
                transitions.append((regimes.iloc[i], regimes.iloc[i + 1]))
    
    stats['actual_transitions'] = {
        'total_count': len(transitions),
        'unique_transitions': len(set(transitions)),
        'transitions_per_period': len(transitions) / len(df) if len(df) > 0 else 0
    }
    
    # Most common transitions
    if transitions:
        transition_counts = pd.Series(transitions).value_counts()
        stats['most_common_transitions'] = [
            {
                'from': t[0],
                'to': t[1],
                'count': count,
                'percentage': count / len(transitions) * 100
            }
            for t, count in transition_counts.head(10).items()
        ]
    
    return stats


def plot_transition_matrix_data(transition_matrix: pd.DataFrame) -> Dict:
    """
    Prepare transition matrix data for plotting.
    
    Args:
        transition_matrix: Transition probability matrix
    
    Returns:
        Dictionary with data formatted for visualization
    """
    return {
        'matrix': transition_matrix.values.tolist(),
        'regimes': transition_matrix.index.tolist(),
        'annotations': transition_matrix.applymap(lambda x: f'{x:.2f}').values.tolist()
    }
