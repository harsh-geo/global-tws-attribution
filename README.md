# Global Terrestrial Water Storage (TWS) Decline Trends & Attribution Pipeline

[![MATLAB](https://img.shields.io/badge/MATLAB-R2021b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![HPC](https://img.shields.io/badge/HPC-DIRAC--IISERK-orange.svg)](https://www.iiserkol.ac.in)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end high-performance computational pipeline for identifying global Terrestrial Water Storage ($TWS$) trends across the 103 largest river basins and attributing their drivers to natural hydroclimate variability vs. human interventions.

---

## 📌 Project Overview

Terrestrial Water Storage ($TWS$) is a key component of the global hydrological cycle. This repository implements a machine learning and hydrologic modeling pipeline to quantify $TWS$ decline trends and perform driver attribution across 103 major river basins worldwide. The central objective is to understand the cause of TWS changes by mapping the spatial variations and computing the mean conditions over each basin.

### Governing Hydrologic Mass Balance
$$\frac{dTWS}{dt} = TWSC = P - ET - Q - (GW_{abs} + SW_{abs})$$

Where:
- $TWS$: Terrestrial Water Storage Anomalies ($\text{cm}$)
- $TWSC$: Terrestrial Water Storage Change ($\text{cm/month}$)
- $P$: Precipitation ($\text{cm/month}$)
- $ET$: Evapotranspiration ($\text{cm/month}$)
- $Q$: Total Runoff / River Discharge ($\text{cm/month}$)
- $GW_{abs}$: Groundwater Abstraction ($\text{cm/month}$)
- $SW_{abs}$: Surface Water Abstraction ($\text{cm/month}$)

---

## 🔬 Key Scientific Findings

- **Widespread Depletion**: **40.8% (42 out of 103 basins)** exhibit statistically significant negative TWS trends ($p < 0.05$, Hamed & Rao Modified Mann-Kendall test), with a mean decline rate of **$-0.528\text{ cm/year}$** among declining basins.
- **Top Depleted Basins**:
  - **Basin 36**: $-1.855\text{ cm/year}$ ($p < 10^{-15}$)
  - **Indus Basin (Basin 51)**: $-1.250\text{ cm/year}$ ($p = 2.68 \times 10^{-8}$)
  - **Basin 14**: $-1.204\text{ cm/year}$ ($p = 1.57 \times 10^{-12}$)
  - **Basin 5**: $-1.171\text{ cm/year}$ ($p = 1.33 \times 10^{-15}$)
  - **Tigris-Euphrates (Basin 17)**: $-0.833\text{ cm/year}$ ($p = 2.82 \times 10^{-5}$)
- **Rigorous Validation Protocol**: Under **3-Year Contiguous Block Cross-Validation** (zero autocorrelation leakage):
  - **82.5% of basins (85/103)** achieve skillful out-of-sample prediction ($\text{NSE} > 0$).
  - **75.7% of basins (78/103)** show improved predictive accuracy under the Full Anthropogenic Model ($M_{anthro} > M_{nat}$), with a **+58.3% increase in median NSE** ($0.0369 \to 0.0584$).
- **Driver Dominance**: Precipitation ($P$) dominates high-frequency meteorological variability across 88.3% of basins, while human groundwater and surface water abstractions drive substantial explanatory gains ($\Delta\text{NSE}$ up to $+0.0894$) across heavily irrigated aquifers.

---

## 🛠️ Data Inputs

All spatial grid variables are formatted to a $0.5^\circ \times 0.5^\circ$ global grid ($720 \times 360$) at a monthly temporal resolution covering April 2002 to December 2019 ($213$ continuous monthly timesteps):

| Dataset | Variable | Raw Source & Units | Standardized Target Unit |
| :--- | :--- | :--- | :--- |
| **GRACE / GRACE-FO** | $TWS$ | JPL/GFZ/CSR Ensemble Mean ($\text{cm}$) | $\text{cm}$ |
| **ERA5** | $P$ | Total Precipitation `tp` ($\text{m/month}$) | $\text{cm/month}$ ($\times 100$) |
| **GLEAM v3.8a** | $ET$ | Evapotranspiration `E` ($\text{mm/month}$) | $\text{cm/month}$ ($/ 10$) |
| **ERA5** | $Q$ | Total Runoff `ro` ($\text{m/month}$) | $\text{cm/month}$ ($\times 100$) |
| **PCR-GLOBWB** | $GW_{abs}$ | Groundwater Abstraction ($\text{m/month}$) | $\text{cm/month}$ ($\times 100$) |
| **PCR-GLOBWB** | $SW_{abs}$ | Surface Water Abstraction ($\text{m/month}$) | $\text{cm/month}$ ($\times 100$) |

---

## 🏗️ Pipeline Architecture

The pipeline processes raw hydroclimate data into publication-ready figures and tables:

### 1. Preprocessing & Data Harmonization (`src/preprocessing/`)
- Unit conversion and anomaly fill value removal (e.g., $-999 \to \text{NaN}$).
- Area-weighted spatial aggregation using latitude-cosine weighting ($\cos(\text{lat})$).
- Explicit MATLAB memory garbage collection (`clear`) for HPC cluster efficiency.

### 2. Random Forest GRACE Gap-Filling (`src/gap_filling/`)
- Reconstructs missing GRACE/GRACE-FO observations (including the 2017–2018 observational gap) using Random Forest regressors trained on historical hydroclimate fluxes ($P, ET, Q, T$).

### 3. Twin Machine Learning Attribution (`src/modeling/`)
- Centered finite difference TWSC calculation: $TWSC(t) = [TWS(t+1) - TWS(t-1)] / (2\Delta t)$.
- **Model 1 ($M_{nat}$)**: Predicts $TWSC$ from natural climate drivers ($P, ET, Q, T, ONI$).
- **Model 2 ($M_{anthro}$)**: Predicts $TWSC$ from natural drivers plus human water abstractions ($GW_{abs}, SW_{abs}$).
- Hyperparameter tuning (`NumPredictorsToSample=1`, deeper trees) and fixed-baseline deseasonalization (2004–2009) to capture weak abstraction signals without temporal leakage.

### 4. Block-Bootstrapping Uncertainty Quantification (`src/modeling/`)
- Non-overlapping 36-month (3-year) Block-Bootstrapping ($N = 1000$ iterations per basin) via `step04c_bootstrap_uncertainty.m`.
- Generates empirical 95% confidence intervals ($2.5^{\text{th}}$ to $97.5^{\text{th}}$ percentiles) for the anthropogenic attribution gain ($\Delta R^2 \pm \text{CI}$), predictor permutation importances, and feature ranking probabilities.

### 5. Validation & Trend Analysis (`src/validation/`)
- 3-Year Contiguous Block Cross-Validation across all 103 basins.
- Autocorrelation-corrected Hamed & Rao Modified Mann-Kendall Test ($p < 0.05$) and Theil-Sen slope estimation.

### 6. Publication Figures (`visualization/`)
- Global TWS trends and stippled significance maps (`plot_basin_trends.m`).
- Global dominant driver maps (`plot_global_attribution_map.m`).
- Model comparison boxplots and lollipop charts (`plot_model_comparison_boxplots.m`, `plot_delta_r2_lollipop.m`).
- High-resolution case study timeseries and scatter plots (`plot_basin_51_analysis.m`).
- Empirical bootstrap confidence intervals & predictor distributions (`plot_bootstrap_uncertainty.m`).

---

## 💻 Running the Pipeline

### Local Execution
```matlab
% In MATLAB Command Window:
run('run_pipeline_local.m');
```
Or via Windows command prompt:
```cmd
run_pipeline_local.bat
```

### HPC Batch Execution (DIRAC Supercomputer / SLURM)
```bash
sbatch slurm/submit_pipeline.sh
```

---

## 📊 Outputs & Artifacts

- `outputs/tables/validation_and_trends.mat`: Master validation and trend results matrix.
- `outputs/tables/attribution_results.mat`: Twin model predictions, feature importances, and 95% confidence intervals.
- `outputs/tables/bootstrap_uncertainty_results.mat`: $N = 1000$ block-bootstrap distribution matrices and empirical confidence bounds.
- `outputs/tables/bootstrap_attribution_uncertainty.csv`: Exported CSV summary table with 95% CIs and driver probabilities.
- `outputs/tables/basin_summary_table.csv`: Exported CSV summary table for all 103 basins.
- `outputs/figures/`: High-resolution figures including global trends, attribution maps, bootstrap uncertainty forest plots, and case studies.
