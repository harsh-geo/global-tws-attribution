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

## 🏗️ Work Completed Thus Far

The pipeline successfully transforms raw hydroclimate data into publication-ready trend and attribution analysis through a highly automated 5-step workflow, along with supporting visualization scripts. Here is the detailed breakdown of the work done till now:

### 1. Preprocessing & Data Harmonization
- **Unit Standardization**: Flux conversions from various datasets have been standardized into liquid water equivalent depth ($\text{cm/month}$). Raw, invalid datasets anomalies were systematically removed (fill value handling such as $-999$ or very large values mapping to `NaN`).
- **Basin-Scale Aggregation**: Extracted spatially-weighted means using a latitude cosine weighting scheme for the 103 major river basins. Memory optimizations (explicit `clear` garbage collections) ensure compatibility with memory-bound clusters like the DIRAC HPC.

### 2. Random Forest GRACE Gap-Filling
- A machine-learning approach is implemented using ensemble Random Forest regressors to predict missing observations spanning the 2017–2018 GRACE to GRACE-FO inter-mission gap. It uses continuous baseline hydroclimate predictors ($P, ET, Q$). 

### 3. Twin Modeling & Attribution
- TWS Change ($TWSC$) computed via centered finite difference over time.
- **Twin Random Forest Models** are implemented natively to perform the attribution logic:
  - **Natural Baseline Model ($M_{nat}$)**: Trained strictly on climate drivers ($P, ET, Q$).
  - **Anthropogenic Model ($M_{anthro}$)**: Trained on natural climate variables plus human abstractions ($GW_{abs}$ and $SW_{abs}$).
- Quantified attribution through the Variance Explained Gain metric ($\Delta R^2$) as well as Random Forest Out-of-Bag permutation feature importance scoring to rank dominant drivers.

### 4. Rigorous Validation and Trend Extraction
- Modeled the trends explicitly employing **3-Year Contiguous Block Cross-Validation** to avoid time-series autocorrelation leakage common to standard K-Fold CV.
- Scored predictions using established hydrology parameters: Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), RMSE, and Pearson $R^2$.
- Long-term TWS variation extracted mathematically using the non-parametric **Theil-Sen's Slope Estimator**.
- Statistical significance obtained dynamically addressing serial dependence using the **Hamed & Rao Modified Mann-Kendall Test** ($p < 0.05$).

### 5. Advanced Visualization Features
We have added fully-automated mapping functionality (`src/visualization/`):
- **Basin Trend Plotting (`plot_basin_trends.m`)**: Synthesizes the outputs into a visual map overlaying TWS decline trends (Sen slope). Uses a blue-red colorbar reflecting increasing or declining stores and elegantly handles statistical significance by overlaying a stippling pattern (dots) on non-significant basins derived from the MK test overlay.
- **Dominant Driver Identification Map (`plot_global_attribution_map.m`)**: Extracts OOB Feature Importance metrics from the Twin Anthropogenic model and categorizes the globe based on the primary variable controlling spatial TWS anomalies.

---

## 💻 Running the Pipeline

### Local Execution
To run the entire pipeline end-to-end locally, you can use either method:

**Method 1: MATLAB Command Window / GUI**
```matlab
run('run_pipeline_local.m');
```

**Method 2: Command Line (Windows Terminal / PowerShell)**
```cmd
run_pipeline_local.bat
```

### Visualization
Once outputs are generated, run the visualization routines in MATLAB:
```matlab
run('visualization/plot_basin_trends.m');
run('visualization/plot_global_attribution_map.m');
```

### HPC Batch Execution (DIRAC Supercomputer / SLURM)
Submit the full pipeline job to the SLURM workload manager:
```bash
sbatch slurm/submit_pipeline.sh
```

---

## 📊 Outputs & Artifacts

Intermediate grids are heavily memory-optimized and bypassed on disk. The output includes:
- `outputs/tables/validation_and_trends.mat`: Final master results containing continuous reconstructed TWS series, twin model metrics ($\Delta R^2$, NSE, KGE, RMSE), feature importance, block CV predictions, and modified Mann-Kendall trend statistics.
- `outputs/tables/basin_summary_table.csv`: Exported publication-grade CSV summary table.
- `outputs/figures/tws_basin_trends.png`: Visual spatial projection mapping of MK significant trends.
- `outputs/figures/tws_dominant_drivers.png`: Dominant physical/human drivers mapping.
