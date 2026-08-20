# High-Impact Enhancements & Methodological Roadmap

This document outlines the strategic methodological enhancements designed to elevate the TWS Attribution workflow to top-tier hydrology journal standards (*Nature Water*, *Water Resources Research*, *GRL*, *Remote Sensing of Environment*).

---

## ✅ Phase 1: Completed & Validated Enhancements

### 1. Mitigation of Cross-Validation Leakage in Deseasonalization
- **Problem in Prior Work (Gap G3):** Standard deseasonalisation computes the monthly climatology from the *entire* time series, including the test period. This means the seasonal mean of each calendar month has already "seen" the test data before the model is evaluated, introducing subtle but systematic information leakage that inflates apparent model skill. Most ML-based hydrological studies fail to account for this.
- **Implementation**: Fixed 2004–2009 climatological baseline period used for anomaly computation across all training and evaluation folds.
- **Impact**: Prevents temporal data leakage and ensures mathematically rigorous out-of-sample cross-validation metrics.

### 2. Forced Feature-Sampling for Weak Signal Isolation
- **Problem in Prior Work (Gap G5):** When standard Random Forest implementations are applied to TWS prediction with both climate and abstraction variables, the algorithm overwhelmingly selects high-variance climate variables ($P$, $ET$) for splitting — because these dominate the total variance of the target. Groundwater abstraction ($GW_{abs}$), which varies by fractions of a cm/month, is systematically ignored during tree construction. As a result, the physically meaningful anthropogenic signal is effectively buried, and standard implementations fail to detect human water use contributions even in basins where pumping is the dominant driver.
- **Implementation**: Hyperparameter tuning restricting split predictor subsampling (`NumPredictorsToSample = 1`) with deeper tree structures (`MinLeafSize = 5`).
- **Impact**: Overcomes the variance magnitude imbalance between massive meteorological fluxes ($P, ET$) and subtle groundwater abstractions ($GW_{abs}$), increasing $M_{anthro}$ predictive skill over $M_{nat}$ in 75.7% of global basins.

### 3. Autocorrelation-Proof Validation Framework
- **Problem in Prior Work (Gap G3):** Standard random K-fold cross-validation randomly shuffles time indices, allowing temporally adjacent observations (e.g., January 2010 in training, February 2010 in testing) to appear in both sets. Because hydrological time series exhibit strong temporal autocorrelation (multi-year droughts, sustained pumping trends), this inflates skill metrics and produces overconfident conclusions about model performance. This is a pervasive problem across the ML-hydrology literature.
- **Implementation**: Replaced standard K-Fold CV with non-overlapping **3-Year Contiguous Block Cross-Validation**.
- **Impact**: Eliminates serial correlation bias, demonstrating robust out-of-sample generalization ($\text{NSE} > 0$ in 82.5% of basins).

### 4. Rigorous Statistical Trend Estimation
- **Problem in Prior Work (Gap G4 + general):** Many TWS trend analyses use standard linear regression or the standard (uncorrected) Mann-Kendall test. The former is sensitive to outliers; the latter does not account for temporal autocorrelation in the data, which inflates the variance of the test statistic and produces spuriously significant trends. Additionally, the ~11-month GRACE–GRACE-FO gap (Jul 2017–May 2018) is often excluded, truncating the time series during a critical drought period and biasing trend estimates.
- **Implementation**: Theil-Sen robust slope estimator coupled with the Hamed & Rao Modified Mann-Kendall test.
- **Impact**: Corrects variance inflation from serial autocorrelation up to lag $n/4$, identifying 42 statistically significant depletion hotspots ($p < 0.05$).

### 5. Multi-Centre Auxiliary Hydroclimate Forcing
- **Problem in Prior Work (Gap G1):** Single-model attribution frameworks are constrained to the forcing variables available within that model's architecture. By relying exclusively on one model's internal variables, important climate teleconnection modes (ENSO, PDO) and thermodynamic controls (temperature-driven ET changes) that modulate storage variability at interannual scales may be missed entirely.
- **Implementation**: Integrated ERA5 2m temperature (`t2m`) and Oceanic Niño Index (ONI) alongside mass balance fluxes ($P, ET, Q, GW_{abs}, SW_{abs}$).
- **Impact**: Captures thermodynamic controls and teleconnection modes modulating storage variability.

### 6. Block-Bootstrapping Uncertainty Quantification ($N = 1000$)
- **Problem in Prior Work (Gaps G1 + G3):** Model-based attribution studies typically report point estimates of driver contributions without uncertainty bounds. This makes it impossible to assess whether the reported attribution is statistically robust or falls within the noise floor. Furthermore, standard bootstrap methods that resample individual time steps violate the temporal dependence structure of the data.
- **Implementation**: Moving / Contiguous Block-Bootstrapping with 36-month (3-year) non-overlapping time blocks ($N_{\text{boot}} = 1000$ iterations per basin) executed in parallel via `step04c_bootstrap_uncertainty.m`.
- **Impact**: Establishes explicit empirical 95% confidence intervals ($2.5^{\text{th}}$ to $97.5^{\text{th}}$ percentiles) for the anthropogenic attribution gain ($\Delta R^2 \pm \text{CI}$), individual feature permutation importance distributions, and driver ranking stability probabilities across all 103 basins.

### 7. Local Explainable AI (XAI)
- **Problem in Prior Work (Gap G2 / G5):** Global feature importance metrics (like OOB permutation importance) describe the average influence of a driver over the entire 20-year study period. However, TWS changes are driven by transient, localized events — a 3-year intense drought, or a 5-year period of severe groundwater over-extraction. Standard global metrics fail to explain *when* human impacts are dominating during these specific extreme events.
- **Implementation**: Integrated SHAP (SHapley Additive exPlanations) to compute localized, monthly-timestep attribution values.
- **Impact**: Enables event-level driver attribution, distinguishing between chronic long-term human depletion and acute climate-driven shocks at a monthly resolution.

---

## 🚀 Phase 2: Future High-Impact Roadmap Extensions

### 1. Temporal Deep Learning Architectures
- **LSTM / Temporal Fusion Transformers (TFT)**: Model long-term physical memory and multi-month antecedent groundwater recharge lags.
- **Physics-Informed Neural Networks (PINNs)**: Embed mass balance conservation ($TWSC = P - ET - Q - \Delta S$) directly into the loss function.

### 2. High-Resolution & Auxiliary Datasets
- **Vegetation Dynamics**: Integrate Leaf Area Index (LAI) or MODIS NDVI to capture dynamic vegetation transpiration.
- **Dam & Reservoir Operations**: Incorporate the Global Reservoir and Dam Database (GRanD) or ResOps to explicitly account for surface water storage buffering.
- **Root-Zone Soil Moisture**: Integrate SMAP or GLDAS soil moisture fields to decouple shallow unsaturated dynamics from deep aquifer drawdown.

### 3. Non-Linear Regime Shift & Breakpoint Analysis
- **Breakpoint Detection**: Implement BFAST or Pettitt's tests to identify abrupt regime shifts caused by episodic megadroughts or major policy shifts in groundwater extraction.

### 4. Input Error Propagation & Quantile Ensembles
- **Input Error Propagation**: Propagate native GRACE harmonic/mascon error fields through the machine learning ensemble.
- **Quantile Regression Forests**: Construct dynamic non-parametric quantile prediction envelopes (5th, 50th, 95th percentiles) for individual basin reconstructions.
