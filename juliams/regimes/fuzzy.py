"""
Fuzzy logic inference system for probabilistic regime classification.
Provides degrees of membership in each regime rather than crisp categories.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import scikit-fuzzy
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    logger.warning("scikit-fuzzy not available. Fuzzy logic features will be limited.")


class FuzzyRegimeClassifier:
    """
    Fuzzy logic system for probabilistic market regime classification.
    
    Instead of hard thresholds, uses fuzzy membership functions to provide
    degrees of membership in each regime.
    """
    
    def __init__(self,
                 trend_range: Tuple[float, float] = (-3.0, 3.0),
                 volatility_range: Tuple[float, float] = (0.0, 0.5),
                 resolution: int = 1001):
        """
        Initialize the fuzzy regime classifier.
        
        Args:
            trend_range: Range for trend strength universe
            volatility_range: Range for volatility universe
            resolution: Number of points in universe discretization
        """
        self.trend_range = trend_range
        self.volatility_range = volatility_range
        self.resolution = resolution
        
        if SKFUZZY_AVAILABLE:
            self._setup_fuzzy_system()
        else:
            self._setup_manual_system()
    
    def _setup_fuzzy_system(self):
        """Set up the fuzzy inference system using scikit-fuzzy."""
        # Define universes
        self.trend_universe = np.linspace(
            self.trend_range[0], self.trend_range[1], self.resolution
        )
        self.vol_universe = np.linspace(
            self.volatility_range[0], self.volatility_range[1], self.resolution
        )
        
        # Define membership functions for trend
        self.trend_down = fuzz.trapmf(
            self.trend_universe, 
            [self.trend_range[0], self.trend_range[0], -0.5, -0.1]
        )
        self.trend_sideways = fuzz.trimf(
            self.trend_universe,
            [-0.3, 0.0, 0.3]
        )
        self.trend_up = fuzz.trapmf(
            self.trend_universe,
            [0.1, 0.5, self.trend_range[1], self.trend_range[1]]
        )
        
        # Define membership functions for volatility
        vol_mid = np.mean(self.volatility_range)
        self.vol_low = fuzz.trapmf(
            self.vol_universe,
            [0, 0, vol_mid * 0.6, vol_mid * 1.2]
        )
        self.vol_high = fuzz.trapmf(
            self.vol_universe,
            [vol_mid * 0.8, vol_mid * 1.4, self.volatility_range[1], self.volatility_range[1]]
        )
    
    def _setup_manual_system(self):
        """Set up a manual fuzzy system when scikit-fuzzy is not available."""
        logger.info("Using manual fuzzy system implementation")
    
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """
        Calculate triangular membership function value.
        
        Args:
            x: Input value
            a, b, c: Triangle parameters (left, peak, right)
        
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    def _trapezoidal_membership(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Calculate trapezoidal membership function value.
        
        Args:
            x: Input value
            a, b, c, d: Trapezoid parameters
        
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)
    
    def get_trend_memberships(self, trend_strength: float) -> Dict[str, float]:
        """
        Calculate fuzzy membership degrees for trend categories.
        
        Args:
            trend_strength: Trend strength value
        
        Returns:
            Dictionary with membership degrees for each trend category
        """
        if SKFUZZY_AVAILABLE:
            # Use scikit-fuzzy interpolation
            down_degree = fuzz.interp_membership(
                self.trend_universe, self.trend_down, trend_strength
            )
            sideways_degree = fuzz.interp_membership(
                self.trend_universe, self.trend_sideways, trend_strength
            )
            up_degree = fuzz.interp_membership(
                self.trend_universe, self.trend_up, trend_strength
            )
        else:
            # Manual calculation
            down_degree = self._trapezoidal_membership(
                trend_strength, self.trend_range[0], self.trend_range[0], -0.5, -0.1
            )
            sideways_degree = self._triangular_membership(
                trend_strength, -0.3, 0.0, 0.3
            )
            up_degree = self._trapezoidal_membership(
                trend_strength, 0.1, 0.5, self.trend_range[1], self.trend_range[1]
            )
        
        return {
            'Down': down_degree,
            'Sideways': sideways_degree,
            'Up': up_degree
        }
    
    def get_volatility_memberships(self, volatility: float) -> Dict[str, float]:
        """
        Calculate fuzzy membership degrees for volatility categories.
        
        Args:
            volatility: Volatility value
        
        Returns:
            Dictionary with membership degrees for each volatility category
        """
        if SKFUZZY_AVAILABLE:
            # Use scikit-fuzzy interpolation
            low_degree = fuzz.interp_membership(
                self.vol_universe, self.vol_low, volatility
            )
            high_degree = fuzz.interp_membership(
                self.vol_universe, self.vol_high, volatility
            )
        else:
            # Manual calculation
            vol_mid = np.mean(self.volatility_range)
            low_degree = self._trapezoidal_membership(
                volatility, 0, 0, vol_mid * 0.6, vol_mid * 1.2
            )
            high_degree = self._trapezoidal_membership(
                volatility, vol_mid * 0.8, vol_mid * 1.4, 
                self.volatility_range[1], self.volatility_range[1]
            )
        
        return {
            'Low': low_degree,
            'High': high_degree
        }
    
    def get_regime_memberships(self, 
                              trend_strength: float,
                              volatility: float,
                              normalize: bool = True) -> Dict[str, float]:
        """
        Calculate fuzzy membership degrees for all six regimes.
        
        Uses minimum (AND) operator for combining trend and volatility memberships.
        
        Args:
            trend_strength: Trend strength value
            volatility: Volatility value
            normalize: Whether to normalize memberships to sum to 1
        
        Returns:
            Dictionary with membership degrees for each regime
        """
        # Check for invalid inputs
        if not np.isfinite(trend_strength) or not np.isfinite(volatility):
            return {regime: 0.0 for regime in [
                'Up-LowVol', 'Up-HighVol', 'Sideways-LowVol', 
                'Sideways-HighVol', 'Down-LowVol', 'Down-HighVol'
            ]}
        
        # Get component memberships
        trend_memberships = self.get_trend_memberships(trend_strength)
        vol_memberships = self.get_volatility_memberships(volatility)
        
        # Combine using minimum (Mamdani inference)
        regime_memberships = {
            'Up-LowVol': min(trend_memberships['Up'], vol_memberships['Low']),
            'Up-HighVol': min(trend_memberships['Up'], vol_memberships['High']),
            'Sideways-LowVol': min(trend_memberships['Sideways'], vol_memberships['Low']),
            'Sideways-HighVol': min(trend_memberships['Sideways'], vol_memberships['High']),
            'Down-LowVol': min(trend_memberships['Down'], vol_memberships['Low']),
            'Down-HighVol': min(trend_memberships['Down'], vol_memberships['High']),
        }
        
        # Normalize if requested
        if normalize:
            total = sum(regime_memberships.values())
            if total > 1e-8:  # Avoid division by very small numbers
                regime_memberships = {
                    k: v / total for k, v in regime_memberships.items()
                }
            else:
                # If all memberships are zero, set uniform distribution
                regime_memberships = {k: 1.0/6.0 for k in regime_memberships.keys()}
        
        return regime_memberships
    
    def classify_fuzzy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform fuzzy classification on entire DataFrame.
        
        Args:
            df: DataFrame with trend_strength and volatility columns
        
        Returns:
            DataFrame with fuzzy membership columns added
        """
        df = df.copy()
        
        # Check required columns
        if 'trend_strength' not in df.columns:
            raise ValueError("DataFrame must contain 'trend_strength' column")
        if 'volatility' not in df.columns:
            raise ValueError("DataFrame must contain 'volatility' column")
        
        # Initialize membership columns
        for regime in ['Up-LowVol', 'Up-HighVol', 'Sideways-LowVol', 
                      'Sideways-HighVol', 'Down-LowVol', 'Down-HighVol']:
            df[f'fuzzy_{regime}'] = 0.0
        
        # Calculate memberships for each row
        for idx in df.index:
            if pd.notna(df.loc[idx, 'trend_strength']) and pd.notna(df.loc[idx, 'volatility']):
                memberships = self.get_regime_memberships(
                    df.loc[idx, 'trend_strength'],
                    df.loc[idx, 'volatility'],
                    normalize=True
                )
                
                for regime, membership in memberships.items():
                    df.loc[idx, f'fuzzy_{regime}'] = membership
        
        # Add primary regime (highest membership)
        fuzzy_cols = [col for col in df.columns if col.startswith('fuzzy_')]
        df['fuzzy_primary_regime'] = df[fuzzy_cols].idxmax(axis=1).str.replace('fuzzy_', '')
        df['fuzzy_confidence'] = df[fuzzy_cols].max(axis=1)
        
        # Add entropy as uncertainty measure
        def calculate_entropy(row):
            probs = row[fuzzy_cols].values
            probs = probs[probs > 1e-10]  # Remove very small values to avoid log issues
            if len(probs) == 0:
                return 0
            probs = np.array(probs, dtype=float)  # Ensure float type
            # Clip probabilities to avoid log(0)
            probs = np.clip(probs, 1e-10, 1.0)
            return -np.sum(probs * np.log(probs))
        
        df['fuzzy_entropy'] = df.apply(calculate_entropy, axis=1)
        
        logger.info(f"Performed fuzzy classification on {len(df)} periods")
        
        return df
    
    def get_fuzzy_nowcast(self, 
                         trend_strength: float,
                         volatility: float) -> Dict:
        """
        Get current fuzzy regime assessment (nowcast).
        
        Args:
            trend_strength: Current trend strength
            volatility: Current volatility
        
        Returns:
            Dictionary with fuzzy nowcast information
        """
        memberships = self.get_regime_memberships(
            trend_strength, volatility, normalize=True
        )
        
        # Find primary regime
        primary_regime = max(memberships, key=memberships.get)
        confidence = memberships[primary_regime]
        
        # Calculate entropy (uncertainty)
        probs = np.array(list(memberships.values()))
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        
        # Aggregate probabilities
        up_prob = memberships['Up-LowVol'] + memberships['Up-HighVol']
        down_prob = memberships['Down-LowVol'] + memberships['Down-HighVol']
        sideways_prob = memberships['Sideways-LowVol'] + memberships['Sideways-HighVol']
        
        high_vol_prob = (memberships['Up-HighVol'] + 
                        memberships['Sideways-HighVol'] + 
                        memberships['Down-HighVol'])
        low_vol_prob = (memberships['Up-LowVol'] + 
                        memberships['Sideways-LowVol'] + 
                        memberships['Down-LowVol'])
        
        nowcast = {
            'memberships': memberships,
            'primary_regime': primary_regime,
            'confidence': confidence,
            'entropy': entropy,
            'trend_probabilities': {
                'Up': up_prob,
                'Sideways': sideways_prob,
                'Down': down_prob
            },
            'volatility_probabilities': {
                'High': high_vol_prob,
                'Low': low_vol_prob
            }
        }
        
        return nowcast
    
    def generate_nowcast_text(self, nowcast: Dict) -> str:
        """
        Generate human-readable text from fuzzy nowcast.
        
        Args:
            nowcast: Nowcast dictionary from get_fuzzy_nowcast
        
        Returns:
            Descriptive text about current market state
        """
        primary = nowcast['primary_regime']
        confidence = nowcast['confidence']
        
        # Map to friendly names
        regime_names = {
            'Up-LowVol': 'Bull Quiet',
            'Up-HighVol': 'Bull Volatile',
            'Sideways-LowVol': 'Sideways Quiet',
            'Sideways-HighVol': 'Sideways Volatile',
            'Down-LowVol': 'Bear Quiet',
            'Down-HighVol': 'Bear Volatile'
        }
        
        primary_name = regime_names.get(primary, primary)
        
        # Determine confidence level
        if confidence > 0.7:
            confidence_text = "high confidence"
        elif confidence > 0.5:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"
        
        # Build text
        text = f"Market is in {primary_name} regime with {confidence_text} ({confidence:.1%}). "
        
        # Add trend assessment
        trend_probs = nowcast['trend_probabilities']
        dominant_trend = max(trend_probs, key=trend_probs.get)
        text += f"Trend direction: {dominant_trend} ({trend_probs[dominant_trend]:.1%} probability). "
        
        # Add volatility assessment
        vol_probs = nowcast['volatility_probabilities']
        if vol_probs['High'] > vol_probs['Low']:
            text += f"Volatility is elevated ({vol_probs['High']:.1%} probability of high volatility)."
        else:
            text += f"Volatility is subdued ({vol_probs['Low']:.1%} probability of low volatility)."
        
        # Add uncertainty note if entropy is high
        if nowcast['entropy'] > 1.5:
            text += " Note: High uncertainty in classification (mixed signals)."
        
        return text


def compute_fuzzy_features(df: pd.DataFrame,
                          config: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute fuzzy regime classification features.
    
    Args:
        df: DataFrame with trend and volatility features
        config: Configuration dictionary
    
    Returns:
        DataFrame with fuzzy features added
    """
    if config is None:
        config = {
            'fuzzy_trend_range': (-3.0, 3.0),
            'fuzzy_vol_range': (0.0, 0.5)
        }
    
    if not SKFUZZY_AVAILABLE:
        logger.warning("scikit-fuzzy not available. Fuzzy features will use fallback implementation.")
    
    classifier = FuzzyRegimeClassifier(
        trend_range=config.get('fuzzy_trend_range', (-3.0, 3.0)),
        volatility_range=config.get('fuzzy_vol_range', (0.0, 0.5))
    )
    
    return classifier.classify_fuzzy(df)


def get_fuzzy_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for fuzzy classification.
    
    Args:
        df: DataFrame with fuzzy classification columns
    
    Returns:
        Dictionary with fuzzy classification summary
    """
    summary = {}
    
    # Check if fuzzy columns exist
    fuzzy_cols = [col for col in df.columns if col.startswith('fuzzy_') and 
                  col not in ['fuzzy_primary_regime', 'fuzzy_confidence', 'fuzzy_entropy']]
    
    if not fuzzy_cols:
        return summary
    
    # Average memberships
    summary['average_memberships'] = {
        col.replace('fuzzy_', ''): df[col].mean()
        for col in fuzzy_cols
    }
    
    # Current memberships
    if len(df) > 0:
        current_memberships = {
            col.replace('fuzzy_', ''): df[col].iloc[-1]
            for col in fuzzy_cols
        }
        summary['current_memberships'] = current_memberships
        
        if 'fuzzy_primary_regime' in df.columns:
            summary['current_primary_regime'] = df['fuzzy_primary_regime'].iloc[-1]
        if 'fuzzy_confidence' in df.columns:
            summary['current_confidence'] = df['fuzzy_confidence'].iloc[-1]
        if 'fuzzy_entropy' in df.columns:
            summary['current_entropy'] = df['fuzzy_entropy'].iloc[-1]
    
    # Statistics on confidence and entropy
    if 'fuzzy_confidence' in df.columns:
        summary['confidence_stats'] = {
            'mean': df['fuzzy_confidence'].mean(),
            'std': df['fuzzy_confidence'].std(),
            'min': df['fuzzy_confidence'].min(),
            'max': df['fuzzy_confidence'].max()
        }
    
    if 'fuzzy_entropy' in df.columns:
        summary['entropy_stats'] = {
            'mean': df['fuzzy_entropy'].mean(),
            'std': df['fuzzy_entropy'].std(),
            'min': df['fuzzy_entropy'].min(),
            'max': df['fuzzy_entropy'].max()
        }
    
    # Primary regime distribution
    if 'fuzzy_primary_regime' in df.columns:
        primary_counts = df['fuzzy_primary_regime'].value_counts()
        total = len(df)
        summary['primary_regime_distribution'] = {
            regime: count / total for regime, count in primary_counts.items()
        }
    
    return summary
