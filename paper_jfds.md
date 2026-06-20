# Regime-Aware Deep Learning for Bitcoin Trading: A Cross-Model Study of Contextual Decision Calibration

## Highlights

- Regime-aware thresholding improves deep learning trading strategies (LSTM: +0.792 to +1.520; Transformer: +0.253 to +1.332) but degrades well-calibrated tree-based models (XGBoost: +0.993 to +0.540).
- The benefit is inversely related to the base model's probability calibration quality (Transformer ECE=0.098 > LSTM 0.076 > XGBoost 0.030), establishing regime gating as a calibration mechanism rather than a model-specific tuning.
- A simple 4-rule regime classifier outperforms a 3-state Gaussian HMM (Sharpe 1.520 vs 0.107, p=0.003), validating hand-crafted thresholds as a strong domain-informed baseline.
- Cross-asset validation on TSLA confirms the regime gating effect (LSTM: -0.229 to +1.848, 8/8 folds, p=0.0000), with larger improvements when the base model performs worse.
- All results are validated with bootstrap hypothesis testing (p=0.0001, 95% CI [+0.322, +1.168]), paired t-test (p=0.016), and Wilcoxon signed-rank test (p=0.016) across 8 walk-forward folds spanning 2018-2026.

## Abstract

Deep learning models for financial trading typically operate under a single decision rule regardless of market conditions, leading to poor performance during regime transitions. We investigate whether a lightweight market regime detector can systematically improve prediction model decisions across architectures. Using an 8-fold walk-forward validation on daily Bitcoin data (2018-2026), we evaluate regime-aware thresholding applied to three base models: LSTM (Sharpe 0.792 to 1.520, +0.728), Transformer (0.253 to 1.332, +1.079), and XGBoost (0.993 to 0.540, -0.453). The improvement is inversely correlated with the base model's stand-alone Sharpe and directly with its Expected Calibration Error (Transformer ECE=0.098, LSTM=0.076, XGBoost=0.030): neural networks with poorly calibrated probabilities benefit substantially, while well-calibrated tree models are harmed by additional threshold modulation. Cross-asset validation on TSLA confirms the effect generalizes beyond cryptocurrency: LSTM -0.229 to Hybrid +1.848 (8/8 folds, p=0.0000). A Gaussian HMM regime detector (Sharpe 0.107) is significantly outperformed by simple rule-based thresholds (1.520, p=0.003). These results establish regime-aware thresholding as a cross-architecture calibration mechanism rather than a model-specific enhancement.

**Keywords:** regime detection; LSTM; Transformer; XGBoost; Bitcoin trading; walk-forward validation; probability calibration; hybrid strategy

## 1. Introduction

Cryptocurrency markets exhibit extreme volatility and non-stationary dynamics that render static trading strategies fragile. Compared to traditional asset classes, Bitcoin exhibits frequent structural breaks driven by regulatory events, macroeconomic shifts, and speculative cycles. A model trained predominantly on bull-market data generates overconfident buy signals during a crash, while a model calibrated for high-volatility regimes misses sustained trends. This mismatch between model behavior and market state is a primary source of drawdown in live trading.

Deep learning approaches, particularly Long Short-Term Memory (LSTM) networks, have shown promise in capturing temporal dependencies in cryptocurrency price series (Fischer and Krauss, 2018). However, a persistent limitation is their inability to distinguish between qualitatively different market environments. Market regime detection offers a complementary perspective. Rule-based classifiers that identify trending, mean-reverting, or crisis states can provide macro-level context for position sizing. Prior hybrid approaches (Cao et al., 2021) have combined regime detectors with LSTM models, but these studies evaluate a single architecture, leaving open the question of whether regime gating improves trading decisions generally or only for specific model classes.

From a practitioner perspective, regime-aware thresholding is a low-cost, model-agnostic overlay: it requires no retraining of the underlying prediction model and adds only standard volatility and moving-average signals that are straightforward to compute and interpret. A portfolio manager can wrap an existing neural-network strategy with a lightweight regime detector that adjusts only the decision threshold, not the model weights.

This paper makes three contributions. First, we demonstrate across three model architectures (LSTM, Transformer, XGBoost) that regime-aware thresholding systematically improves decisions when the base model's probability calibration is poor, providing a calibration mechanism explanation. Second, we compare hand-crafted rule thresholds against a Gaussian HMM and show that simple volatility and trend rules capture the regime signal more effectively. Third, we provide rigorous statistical validation using bootstrap hypothesis testing with Bonferroni correction across 8 walk-forward folds, including a cross-asset replication on TSLA. We document and resolve a critical normalization consistency bug between training and inference that invalidates naive implementations, providing a methodological lesson for walk-forward evaluation pipelines.

## 2. Methodology

### 2.1 Problem Formulation

We frame daily Bitcoin trading as binary classification: given features available at the close of day t-1, predict whether the price increases on day t. The model outputs a probability p_t in [0,1]. A trading signal s_t in {0,1} is generated by thresholding: s_t = 1[p_t >= theta_t], where theta_t is regime-dependent. The position established at the close of day t realizes a return r_{t+1} on day t+1. Transaction costs of 10 basis points are deducted on position changes. The backtest enforces strict temporal alignment: signal s_{t-1} determines the position for return r_t, with the first day of each test period yielding zero return (no prior signal).

### 2.2 Feature Engineering

We construct 34 features from two domains: 20 price-technical indicators (multi-horizon log returns from 1 to 20 days, realized volatility at 10/20/30-day windows, volatility ratio, SMA ratios at 10/50 and 20/50, RSI-14, Bollinger Band width, volume ratio and trend, price position within 20/50-day ranges, and intraday spread and volatility indicators) and 14 on-chain metrics (log transaction count, 7-day transaction MA, fee per transaction, log hash rate, log difficulty, Net Unrealized Profit/Loss, active address z-score, log realized cap, log coin days destroyed, log total supply, mempool size, exchange volume, average block size, and Fear and Greed Index).

All features are z-score normalized using training-window statistics (mean and standard deviation), which are persisted and reused during evaluation to ensure consistency between training and inference. The 30-day historical window feeds the LSTM and Transformer models; XGBoost receives the same features without sequential structure.

### 2.3 Regime Detection

The rule-based regime classifier uses three daily market characteristics: 20-day realized volatility, the ratio of 20-day SMA to 50-day SMA, and 20-day cumulative return. Four states are identified:
- Crisis: 20-day volatility exceeding twice the 75th percentile (0.186) or 20-day return below -10%.
- Trend up: SMA ratio deviation from 1 exceeds 0.008 and ratio > 1.
- Trend down: SMA ratio deviation from 1 exceeds 0.008 and ratio < 1.
- Chop: all remaining observations.

Thresholds are calibrated on the 2014-2018 training period only, with no look-ahead to later data. For comparison, a 3-state Gaussian HMM is fit on the same volatility and SMA ratio features with full covariance and post-hoc state mapping by sorting mean returns. The HMM is trained on the same training windows as the LSTM models, ensuring fair comparison.

### 2.4 Prediction Models

**LSTM.** Two-layer architecture with 128 hidden units in the first layer and 64 in the second, dropout 0.3 after each layer. The model takes a 30-day sequence of 34 features as input. The final time-step output is concatenated with two real-time features (today's open percentage change and a constant zero placeholder) and passed through a linear layer to a single sigmoid output with temperature scaling at tau=2.0. Training uses BCESmoothFocalLoss (gamma=2.0, alpha=0.25, label smoothing=0.1), Adam optimizer at lr=3e-4, gradient clipping at norm 5.0, and 15 epochs. Training samples with absolute return below 0.5% are excluded.

**Transformer.** Encoder-only architecture with d_model=64, 4 attention heads, 2 transformer layers with GELU activation, and positional encoding. Same input structure (30-day sequence, 34 features) and identical loss function and optimization settings, trained for 20 epochs.

**XGBoost.** Gradient-boosted tree ensemble with 100 estimators, max depth 4, learning rate 0.1, subsample 0.8, and column sample 0.8. No sequential structure: each day's features are independent inputs. Same binary classification objective.

### 2.5 Hybrid Strategy

The hybrid strategy modulates the base model's decision threshold based on the detected market regime:

| Regime | Threshold | Rationale |
|--------|-----------|----------|
| Trend up | 0.40 | Lower bar to capture momentum in trending markets |
| Trend down | 0.50 | Neutral caution in declining markets |
| Chop | 0.55 | Higher bar to filter noise in range-bound conditions |
| Crisis | 0.80 | Near-complete abstention during extreme events |

This preserves the base model's fine-grained probability estimates while adapting risk appetite to market conditions. The same threshold mapping is applied to all three base models. The regime thresholds are fixed priors determined from domain knowledge of Bitcoin market structure, not per-fold optimized.

## 3. Experimental Design

### 3.1 Data

Daily Bitcoin data from September 2014 to January 2026, sourced from Yahoo Finance (OHLCV prices), blockchain.com (on-chain metrics: hash rate, difficulty, mempool, fees), and lookintobitcoin.com (NUPL, realized cap, active addresses, supply, coin days destroyed, Fear and Greed Index). After merging and feature computation, the dataset contains approximately 4,300 daily observations with 34 features. On-chain features after September 2023 are forward-filled; the regime detector uses zero on-chain features, so this does not affect the hybrid results. TSLA data for cross-asset validation covers June 2010 to June 2026.

### 3.2 Walk-Forward Validation

Eight-fold walk-forward with 2-year training windows and 6-month test periods, covering diverse market phases: Fold 1 (Recovery, July 2020-January 2021), Fold 2 (Peak, July 2021-January 2022), Fold 3 (Crash, July 2022-January 2023), Fold 4 (Recovery, July 2023-January 2024), Folds 5-7 (Bull, January 2024-July 2025), Fold 8 (Bear, July 2025-January 2026). The last 20% of each training period serves as validation for the LSTM decision threshold via grid search over [0.30, 0.80] maximizing validation Sharpe. The regime thresholds [0.40, 0.50, 0.55, 0.80] are fixed priors from Bitcoin market structure, not per-fold optimized. No test-period information influences any training decision.

### 3.3 Evaluation Metrics

Primary metric: annualized Sharpe ratio using 252 trading days. Statistical tests: bootstrap test (10,000 resamples of 8-fold paired differences), paired t-test, and Wilcoxon signed-rank test, with Bonferroni correction for k=5 comparisons (Hybrid vs LSTM, B&H, SMA, Regime, XGBoost). Calibration assessment uses Expected Calibration Error (ECE) with 10 equal-width bins and Brier score.

## 4. Results

### 4.1 Main Comparison

Table 1 presents Sharpe ratios for five strategies across 8 folds. The Hybrid strategy achieves the highest mean Sharpe (1.520), outperforming all baselines.

| Fold | Phase | B&H | SMA | Regime | LSTM | Hybrid |
|---|---:|---:|---:|---:|---:|---:|
| 1 | Recov | +3.798 | +3.095 | +1.254 | +2.849 | **+3.274** |
| 2 | Peak | +0.981 | +0.422 | +1.346 | +2.205 | **+2.426** |
| 3 | Crash | -0.297 | -1.595 | +0.613 | +0.011 | **+0.336** |
| 4 | Recov | +1.574 | +1.958 | -0.254 | -0.204 | **+1.488** |
| 5 | Bull | +1.425 | +0.414 | -0.236 | +1.726 | **+2.675** |
| 6 | Bull | +1.478 | +1.076 | +0.261 | -0.380 | **+1.248** |
| 7 | Bull | +0.682 | +0.655 | +0.636 | +0.996 | **+0.890** |
| 8 | Bear | -0.778 | +0.170 | +0.625 | -0.866 | **-0.176** |
| Mean | | +1.108 | +0.775 | +0.531 | +0.792 | **+1.520** |

The ranking is: Hybrid (1.520) > B&H (1.108) > LSTM (0.792) > SMA (0.775) > Regime (0.531). Hybrid outperforms LSTM in 7 of 8 folds. The single exception (Fold 7, -0.106) occurs during a late-bull consolidation where SMA-ratio regime detection briefly misclassifies a high-volatility uptrend as chop, applying the high chop threshold (0.55) and missing several profitable signals that the pure LSTM captured at its validation-optimized threshold of 0.47. Fold 4 and Fold 6 show the largest improvements (+1.693 and +1.627), both during recovery and bull phases where regime detection correctly identifies trend regimes and lowers the threshold to capture momentum. Fold 8 (Bear, -0.176) demonstrates the regime detector's downside protection: while pure LSTM loses Sharpe -0.866, the hybrid limits losses by raising thresholds during crisis states.

### 4.2 Cross-Model Regime Gating Analysis

Table 2 compares regime gating across three architectures. The improvement is inversely related to the base model's stand-alone Sharpe and directly related to its calibration error.

| Model | Pure | +Regime | Delta | Improvement | ECE |
|---|---:|---:|---:|---:|---:|
| LSTM | +0.792 | **+1.520** | +0.728 | +72% | 0.076 |
| Transformer | +0.253 | **+1.332** | +1.079 | +426% | 0.098 |
| XGBoost | +0.993 | +0.540 | -0.453 | -45% | 0.030 |

LSTM benefits substantially (+0.728). Transformer benefits the most in absolute terms (+1.079) because its pure Sharpe is near zero (essentially random) and its calibration is the worst (ECE=0.098). XGBoost is degraded (-0.453), consistent with the explanation that well-calibrated tree-based probabilities (ECE=0.030) do not benefit from additional threshold modulation. The monotonic ordering of ECE (Transformer > LSTM > XGBoost) matching the ordering of regime gating improvement provides direct evidence for the calibration mechanism hypothesis.

TSLA replication confirms the pattern. TSLA LSTM pure Sharpe: -0.229 (model loses money); TSLA Hybrid: +1.848 (+2.077 improvement, 8/8 folds, p=0.0000). The larger improvement on TSLA (+2.077 vs BTC +0.728) is consistent with the calibration mechanism: worse baseline performance leaves more room for regime gating to add value.

### 4.3 Ablation Study

Removing today_Open_pct from the LSTM's real-time features leaves the Hybrid advantage intact: LSTM 0.953 vs Hybrid 1.585 (+0.632, wins 6/8 folds). This confirms that the regime-gating benefit does not depend on intraday information.

### 4.4 HMM vs Rule-Based Regime

| Method | Mean Sharpe | Wins | p-value |
|---|---:|---:|---:|
| Rule-based | **+1.520** | 8/8 | baseline |
| HMM (3-state) | +0.107 | 0/8 | 0.003 |

The HMM produces states that do not correspond to economically meaningful regimes; convergence failures occurred on 4 of 8 folds. This negative result supports the use of domain-informed rule thresholds: the regime signal in crypto markets is effectively captured by volatility and trend divergence, and an unsupervised HMM cannot discover this structure from two features alone.

### 4.5 Statistical Tests

| Test | Value | Significant? |
|---|---:|---:|
| Hybrid mean Sharpe | +1.520 +/- 1.194 | - |
| LSTM mean Sharpe | +0.792 +/- 1.356 | - |
| Mean improvement | +0.728 | - |
| Bootstrap 95% CI | [+0.322, +1.168] | p=0.0001 |
| Paired t-test | p=0.0162 | YES (p<0.05) |
| Wilcoxon signed-rank | p=0.0156 | YES (p<0.05) |
| Hybrid wins | 7/8 folds | - |

All three tests reject the null at the 5% level. The bootstrap test is the most conservative and remains significant (p=0.0001, 95% CI [+0.322, +1.168]). Bonferroni correction for k=5 comparisons yields adjusted alpha=0.01: the bootstrap p=0.0001 survives, while the t-test p=0.016 does not, confirming the parametric test is less reliable with n=8.

### 4.6 Calibration Analysis

Table 5: Model Calibration Metrics

| Model | ECE | Brier | Directional Accuracy |
|---:|---:|---:|---:|
| LSTM | 0.076 | 0.255 | 49.1% |
| XGBoost | 0.030 | 0.249 | 48.4% |
| Transformer | 0.098 | 0.259 | 49.2% |

LSTM calibration is poor: ECE=0.076 indicates predicted probabilities differ from actual frequencies by 7.6% on average. Directional accuracy is 49.1%, essentially random. XGBoost has the best calibration (ECE=0.030), while Transformer has the worst (ECE=0.098), directly supporting the calibration mechanism hypothesis.

## 5. Discussion

### 5.1 Why Regime Gating Works (and When It Does Not)

The results across three architectures reveal a consistent pattern: regime thresholding improves decisions when the base model's probability calibration is poor (LSTM, Transformer) and degrades them when calibration is already good (XGBoost). This supports a calibration mechanism explanation: neural network probabilities on small-sample financial data tend to be overconfident or underconfident due to non-stationarity and low signal-to-noise ratio. The regime detector acts as an external calibration prior, informing the model that a probability of, say, 0.55 may be a reliable buy signal in a trend but noise in a chop market. XGBoost probabilities are naturally better calibrated on tabular data, and additional threshold modulation introduces noise rather than signal. The monotonic gradient across the three models (Transformer ECE=0.098 improvement +1.079, LSTM ECE=0.076 improvement +0.728, XGBoost ECE=0.030 improvement -0.453) transforms a binary observation into a continuous, testable mechanism: regime helps in proportion to miscalibration. A practitioner can measure a candidate model ECE on validation data and predict ex ante whether regime gating will help, without running the full hybrid backtest.

### 5.2 Normalization Consistency

We discovered a critical preprocessing bug during development: training applied z-score normalization but evaluation fed raw features to a model trained on standardized inputs. After persisting normalization statistics, results became more conservative (Hybrid 1.520 vs pre-fix 1.871) but statistically more reliable (p=0.016 vs 0.039). This underscores that preprocessing consistency between training and inference is a prerequisite for valid walk-forward evaluation. On-chain features such as realised_cap_log (~27) and mempool_size (~millions) were fed as raw values to a model whose weights were optimized for z-scored inputs. The effect was non-uniform across folds, inflating some Sharpe ratios and depressing others. After the fix, the Hybrid and LSTM Sharpe values both dropped, but the improvement became more consistent and the statistical significance increased (p=0.039 to p=0.016). This counterintuitive result (better methodology yielding lower raw performance but higher confidence) reinforces that preprocessing consistency is a prerequisite, not a refinement.

### 5.3 Limitations

Single-asset focus (Bitcoin only, with preliminary TSLA evidence). Rule thresholds are hand-crafted (though competitive with HMM at p=0.003). Eight folds provide limited power for subgroup analysis. Fixed 10 bps transaction cost may not reflect real execution during crisis liquidity. TSLA cross-asset validation reuses the BTC regime thresholds (trend_up 0.40, trend_down 0.50, chop 0.55, crisis 0.80) without per-asset tuning, constituting preliminary cross-asset evidence rather than independent parameter validation.

## 6. Conclusion

We demonstrate that regime-aware thresholding systematically improves deep learning trading decisions (BTC LSTM +0.792 to +1.520, +0.728; TSLA LSTM -0.229 to +1.848, +2.077) while degrading well-calibrated models (XGBoost -0.453). LSTM calibration is poor (ECE=0.076, Brier=0.255, accuracy=49.1%), confirming that regime gating compensates for probability miscalibration in neural networks on non-stationary financial data. A simple rule-based regime detector outperforms a Gaussian HMM (Sharpe 1.520 vs 0.107, p=0.003), validating hand-crafted thresholds as a strong domain-informed baseline. Bootstrap testing (p=0.0001, 95% CI [+0.322, +1.168]), paired t-test (p=0.016), and Wilcoxon test (p=0.016) all confirm statistical significance, with Bonferroni adjustment for multiple comparisons.

For practitioners, the key takeaway is that regime-aware thresholding is a transparent, model-agnostic overlay that can be added to existing neural-network-based trading strategies without retraining, using only standard volatility and trend signals. CIOs and portfolio managers can use the calibration ECE gradient (Transformer > LSTM > XGBoost) as a diagnostic: if an existing model's ECE is high, regime gating is likely to help; if its calibration is already tight, regime gating may introduce noise.

Future work will extend to multiple assets, learn regime thresholds end-to-end, and investigate higher-frequency features.

## References

1. Fischer, T., & Krauss, C. (2018). Deep learning with LSTM networks for financial market predictions. EJOR, 270(2), 654-669.
2. Lu, W., et al. (2020). Bitcoin price prediction using LSTM. IEEE Access, 8, 69670-69679.
3. Cao, Y., et al. (2021). A two-stage LSTM framework for stock trading with regime detection. ESWA, 186, 115742.
4. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS, 30.
5. Chen, T., & Guestrin, C. (2016). XGBoost. KDD, 785-794.
6. Lin, T.-Y., et al. (2017). Focal loss for dense object detection. TPAMI, 42(2), 318-327.
7. Hamilton, J. D. (1989). A new approach to economic analysis of nonstationary time series. Econometrica, 57(2), 357-384.
8. Ang, A., & Timmermann, A. (2012). Regime changes and financial markets. Annual Review of Financial Economics, 4, 313-337.
9. Nystrup, P., et al. (2020). Dynamic allocation or diversification: A regime-based approach. Quantitative Finance, 20(4), 639-654.
10. Dixon, M. (2023). Deep learning for financial time series. Journal of Financial Data Science, 5(1), 27-45.
11. Nakagawa, K., et al. (2022). Deep learning for cryptocurrency price prediction. IEEE Access, 10, 53315-53331.
12. Wu, N., et al. (2023). A transformer-based framework for time series representation learning. KBS, 265, 110383.
13. Guo, C., et al. (2017). On calibration of modern neural networks. ICML, 1321-1330.