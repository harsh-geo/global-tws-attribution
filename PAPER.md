# Research Paper Outline

**Tentative Title:** 
*Disentangling Natural and Anthropogenic Drivers of Global Terrestrial Water Storage Decline using Machine Learning*

## 1. Abstract (approx. 250 words)
- **Background**: Terrestrial Water Storage (TWS) is depleting rapidly globally, threatening water security.
- **Problem**: Separating natural climate variability (droughts) from human interventions (groundwater/surface water abstraction) remains challenging.
- **Method**: We employ a Twin Machine Learning framework (Random Forest) coupled with 20 years of GRACE satellite data and hydroclimate modeling across the world's 103 largest river basins.
- **Key Findings**: State the percentage of basins significantly declining, and the dominant driver (climate vs. human) across major stressed aquifers.
- **Significance**: Highlights where immediate water policy interventions are required.

## 2. Introduction

### 2.1. Overview: The Global Freshwater Crisis
- **The role of TWS**: Freshwater sustains 8 billion people, irrigates cropland producing >40% of global food, and maintains ecosystems. Terrestrial Water Storage (TWS) — the vertically integrated sum of surface water, soil moisture, groundwater, and snow/ice — is the ultimate finite reservoir upon which all these demands draw.
- **The crisis**: Over the past two decades, TWS has been declining at alarming rates in many of the world's most productive and densely populated river basins, threatening water security, food production, and ecosystem sustainability on a planetary scale (Rodell et al., 2018; Scanlon et al., 2018; Famiglietti, 2014).
- **The attribution problem**: Observed TWS declines may arise from natural hydroclimate variability (droughts, warming-enhanced ET) or anthropogenic water abstractions (groundwater pumping, surface water diversion). These drivers operate simultaneously and interact nonlinearly, making their separation — the *attribution problem* — one of the most challenging questions in contemporary hydrology.

### 2.2. Present Work in the Field (Literature Review)
- **Stream 1 — Satellite Gravimetry**: GRACE (2002–2017) and GRACE-FO (2018–present) have revolutionised monitoring of large-scale TWS changes from space (Tapley et al., 2004). Rodell et al. (2018) documented global freshwater declines; Scanlon et al. (2018) showed global models underestimate observed trends.
- **Stream 2 — Regional Case Studies**: Individual basin studies have documented severe groundwater depletion:
  - Indo-Gangetic Plain: Tiwari et al. (2009), Rodell et al. (2009) — intensive rice-wheat irrigation.
  - California Central Valley: Famiglietti et al. (2011) — rapid aquifer drawdown.
  - Tigris-Euphrates: Voss et al. (2013), Joodaki et al. (2014) — dam construction + unregulated pumping.
  - North China Plain: Feng et al. (2013) — industrial + agricultural abstractions.
- **Stream 3 — Model-Based & Statistical Attribution**:
  - Physics-based models: Döll et al. (2014) — compared model runs with/without human water use against GRACE.
  - Data assimilation: Zaitchik et al. (2008) — assimilated GRACE into a land surface model.
  - Statistical decomposition: Humphrey et al. (2017) — reconstructed climate-driven TWS variability to separate residual anthropogenic signals.

### 2.3. Research Gaps in Existing Approaches
- **G1 — Single-model structural uncertainty**: Most attribution studies depend on one hydrological model (VIC, PCR-GLOBWB, WGHM), whose structural assumptions (infiltration parameterisation, aquifer geometry) propagate unconstrained biases. No independent data-driven verification is provided.
- **G2 — No systematic global-scale driver separation**: Regional case studies cover individual basins; no study performs basin-by-basin climate-vs.-human attribution across all major global basins simultaneously, preventing cross-regime comparison.
- **G3 — Temporal autocorrelation ignored in validation**: ML-based hydrological studies commonly use random K-fold CV, which leaks temporally correlated information (adjacent drought months in train+test), inflating skill metrics.
- **G4 — GRACE observational gap unaddressed**: The ~11-month gap (Jul 2017–May 2018) between GRACE and GRACE-FO is often excluded or linearly interpolated, truncating trend analysis during a critical drought period.
- **G5 — Weak anthropogenic signals masked by climate variance**: Standard ML implementations split on high-variance climate variables ($P$, $ET$), systematically burying the subtle groundwater abstraction signals ($GW_{abs}$, $SW_{abs}$) that vary by fractions of a cm/month.

### 2.4. Objectives & Contributions of This Study
Our work directly addresses Gaps G1–G5 through three methodological pillars:
1. **Gap-fill and reconstruct continuous TWS globally (→ G4)**: RF-based covariate-driven reconstruction of the GRACE–GRACE-FO gap, producing a continuous 213-month record across 103 basins.
2. **Map spatial TWS decline trends robustly (→ G3)**: Theil-Sen slope + Hamed & Rao autocorrelation-corrected Modified Mann-Kendall significance testing.
3. **Attribute drivers using twin ML methodology (→ G1, G2, G5)**:
   - Twin RF framework ($M_{nat}$ vs. $M_{anthro}$) — data-driven, no dependence on a single hydrological model's structure (G1).
   - Systematic application across all 103 largest global basins (G2).
   - Forced feature-sampling (`NumPredictorsToSample = 1`) to amplify weak anthropogenic signals (G5).
   - 3-Year Contiguous Block CV + fixed-baseline deseasonalisation to prevent all temporal leakage (G3).

## 3. Data and Methods
### 3.1. Datasets
- **GRACE/GRACE-FO**: Ensemble mean of JPL, GFZ, CSR.
- **Hydroclimate Drivers**: Precipitation (ERA5), Evapotranspiration (GLEAM), Runoff (ERA5).
- **Human Abstraction**: Groundwater and Surface Water abstraction rates from PCR-GLOBWB.
- **Spatial Domain**: 103 major river basins, aggregated using latitude-cosine weighting.

### 3.2. Machine Learning Gap-Filling
- Reconstructing the 2017–2018 GRACE observational gap using Random Forest regressors trained on historical hydroclimate baseline data.

### 3.3. Trend and Significance Analysis
- Calculation of TWS trends using the robust Theil-Sen’s slope estimator.
- Correcting for temporal autocorrelation using the Modified Mann-Kendall test (Hamed & Rao).

### 3.4. Twin Attribution Framework
- **Model 1 ($M_{nat}$)**: TWS Change driven purely by natural fluxes ($P, ET, Q$).
- **Model 2 ($M_{anthro}$)**: TWS Change driven by natural fluxes + human abstractions.
- **Attribution Metrics**: Variance Explained Gain ($\Delta R^2$) and Out-of-Bag permutation feature importance.

## 4. Results
### 4.1. Global Patterns of TWS Decline
- Visuals: Global map of TWS trends (Stippled for significance).
- Identification of the most severely depleted basins.

### 4.2. Validation of Gap-Filling and Machine Learning Models
- Report performance metrics: NSE, KGE, and Block Cross-Validation RMSE to prove models did not overfit.

### 4.3. Disentangling the Drivers
- Visuals: Global map of dominant drivers (categorized by $P, ET, Q, GW, SW$).
- Scatter plots or bar charts comparing $\Delta R^2$ across the most stressed basins.
- Feature importance analysis highlighting where Human Abstraction severely overrides natural climate signals.

## 5. Discussion
### 5.1. Hotspots of Human-Induced Depletion
- Deep dive into 2-3 specific basins (e.g., Indus, Tigris-Euphrates, Colorado) where $GW_{abs}$ is identified as the dominant driver.

### 5.2. Climate-Driven Depletion
- Discussion of basins where precipitation deficits (long-term droughts) or $ET$ increases (warming) are driving the TWS loss.

### 5.3. Uncertainties and Limitations
- Resolution limits of GRACE.
- Uncertainties inherent in global hydrological models (PCR-GLOBWB) and reanalysis data (ERA5).

## 6. Conclusion
- Summary of the primary findings.
- Policy implications for sustainable groundwater management and climate adaptation.

## 7. References
- *Standard citations (Tapley et al., Rodell et al., Scanlon et al., etc.)*
