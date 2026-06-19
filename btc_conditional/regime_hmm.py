"""HMM-based regime detector — learnable alternative to rule-based thresholds"""

import numpy as np
from hmmlearn import hmm
import pandas as pd

class HMMRegimeDetector:
    """HMM-based market regime detector.
    
    Uses 2 observable features (volatility_20d, sma_ratio_20_50) 
    to infer hidden market states via Gaussian HMM.
    
    States are post-hoc labeled: given historical return distributions
    per state, the state with highest mean return → trend_up, etc.
    """
    
    def __init__(self, n_states=3, random_state=42):
        self.n_states = n_states
        self.model = None
        self.state_map = {}  # HMM state ID → regime label
    
    def _extract_features(self, df):
        """Extract regime-relevant features from full dataframe."""
        features = pd.DataFrame({
            'volatility_20d': df['volatility_20d'].values,
            'sma_ratio_20_50': df['sma_ratio_20_50'].values,
        })
        # Handle NaNs (first 50 days of rolling)
        features = features.bfill().fillna(0)
        return features.values
    
    def fit(self, df, ret_col='return_1d'):
        """Fit HMM on training data, then label states by return characteristics.
        
        df: full dataframe  (the method will use only the training portion later)
        """
        X = self._extract_features(df)
        
        # Fit HMM with 2D Gaussian emissions
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            random_state=42,
            n_iter=100,
            tol=1e-4
        )
        self.model.fit(X)
        
        # Decode hidden states
        hidden_states = self.model.predict(X)
        
        # Label states: compute mean return per state, then assign regime labels
        returns = df[ret_col].fillna(0).values
        state_returns = {}
        for s in range(self.n_states):
            mask = hidden_states == s
            state_returns[s] = float(np.mean(returns[mask]))
        
        # Sort states by mean return
        sorted_states = sorted(state_returns, key=state_returns.get)
        
        # Map: worst → trend_down, middle → chop, best → trend_up
        labels = ['trend_down', 'chop', 'trend_up']
        if self.n_states == 2:
            labels = ['trend_down', 'trend_up']
        
        for i, state in enumerate(sorted_states):
            label = labels[i] if i < len(labels) else 'chop'
            self.state_map[state] = label
        
        return self
    
    def predict(self, df, idx=None):
        """Predict regime for a single row or slice."""
        if idx is not None:
            row = df.iloc[idx]
        else:
            row = df
        
        vol = row['volatility_20d']
        sr = row['sma_ratio_20_50']
        
        if np.isnan(vol) or np.isnan(sr):
            return 'chop'
        
        X = np.array([[vol, sr]])
        state = self.model.predict(X)[0]
        return self.state_map.get(state, 'chop')
    
    def predict_batch(self, df, start_idx=0):
        """Predict regimes for a contiguous batch."""
        sub = df.iloc[start_idx:].copy()
        X = self._extract_features(sub)
        states = self.model.predict(X)
        return [self.state_map.get(s, 'chop') for s in states]
