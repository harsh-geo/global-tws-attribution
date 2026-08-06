# Global Terrestrial Water Storage (TWS) Decline Trends & Attribution Pipeline

[![MATLAB](https://img.shields.io/badge/MATLAB-R2021b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![HPC](https://img.shields.io/badge/HPC-DIRAC--IISERK-orange.svg)](https://www.iiserkol.ac.in)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end high-performance computational pipeline for identifying global Terrestrial Water Storage ($TWS$) trends across the 103 largest river basins and attributing their drivers to natural hydroclimate variability vs. human interventions.

---

## 📌 Project Overview

Terrestrial Water Storage ($TWS$) is a key component of the global hydrological cycle. This repository implements a machine learning and hydrologic modeling pipeline to quantify $TWS$ decline trends and perform driver attribution across 103 major river basins worldwide.

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

## 🏗️ Directory Architecture

```
data2/
├── GEMINI.md                            # Agent core context & HPC execution directives
├── README.md                            # Project documentation (this file)
├── grace/                               # GRACE NetCDF datasets & dates
│   ├── grace_corrected_monthly.nc
│   └── grace_dates.mat
├── data/
│   ├── raw/                             # Raw input files & masks
│   └── processed/                       # Mask grid & raw input files (intermediate .mat saves disabled)
│       └── basin_map.mat                # 103 major river basin mask grid
├── src/
│   ├── preprocessing/
│   │   ├── step01_unit_conversion.m    # Unit standardization & timeline alignment
│   │   └── step02_aggregate_basins.m    # Latitude cosine-weighted spatial aggregation
│   ├── gap_filling/
│   │   └── step03_reconstruct_grace.m   # RF gap-filling for missing GRACE months
│   ├── modeling/
│   │   └── step04_run_attribution.m    # Twin Random Forest attribution modeling
│   └── validation/
│       └── step05_validate_and_trends.m # 3-Yr block cross-validation & Mann-Kendall trends
├── visualization/
│   ├── plot_basin_trends.m              # Basin-level trend plotting
│   └── plot_global_attribution_map.m    # Global attribution map generation
└── slurm/
    ├── submit_pipeline.sh               # DIRAC HPC batch SLURM job submitter
    └── logs/                            # SLURM output & error logs
```

---

## 🚀 Pipeline Workflow

### Step 1: Hydroclimate Flux Standardization & Reindexing
`src/preprocessing/step01_unit_conversion.m`
- Standardizes all hydroclimate flux inputs to liquid water equivalent depth ($\text{cm/month}$).
- Sanitizes non-physical fill values ($-999$, $>10^{19}$) to standard MATLAB `NaN`s.
- Maps GRACE observations onto the 213-month timeline (April 2002 – December 2019), placing `NaN`s at unobserved gap months.

### Step 2: Latitude Cosine-Weighted Basin Aggregation
`src/preprocessing/step02_aggregate_basins.m`
- Reads `basin_map.mat` (IDs $1\dots103$).
- Applies latitude cosine-weighting ($\cos(\text{lat})$) across the $0.5^\circ$ grid to extract spatial weighted averages for each basin:
  $$\bar{x}_{t, b} = \frac{\sum_{i \in \text{basin } b} x_{i, t} \cdot \cos(\text{lat}_i)}{\sum_{i \in \text{basin } b} \cos(\text{lat}_i)}$$

### Step 3: Random Forest GRACE Gap-Filling
`src/gap_filling/step03_reconstruct_grace.m`
- Reconstructs missing GRACE/GRACE-FO observation months (including the 2017–2018 inter-mission gap).
- Trains an ensemble Random Forest regressor (`TreeBagger`) per basin using hydroclimate predictors ($P, ET, Q$) during continuous baseline observation periods (2002–2017).

### Step 4: Twin Attribution Modeling
`src/modeling/step04_run_attribution.m`
- Calculates $TWSC$ via centered finite differences:
  $$TWSC(t) = \frac{TWS(t+1) - TWS(t-1)}{2 \Delta t}$$
- Trains twin models:
  - **Natural Baseline Model ($M_{nat}$)**: Predicts $TWSC$ using $P, ET, Q$.
  - **Full Anthropogenic Model ($M_{anthro}$)**: Predicts $TWSC$ using $P, ET, Q, GW_{abs}, SW_{abs}$.
- Computes variance explained gain ($\Delta R^2 = R^2_{anthro} - R^2_{nat}$) and Out-of-Bag Permutation Feature Importance.

### Step 5: Validation & Trend Analysis
`src/validation/step05_validate_and_trends.m`
- Implements **3-Year Contiguous Block Cross-Validation** to prevent temporal autocorrelation leakage.
- Evaluates model performance using Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), RMSE, and Pearson $R^2$.
- Computes trend magnitudes using **Theil-Sen's Slope Estimator** ($\text{cm/yr}$) and assesses statistical significance ($p < 0.05$) via the **Hamed and Rao Modified Mann-Kendall Test** (autocorrelation-corrected).

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
*or via MATLAB batch mode:*
```bash
matlab -batch "run('run_pipeline_local.m');"
```

### HPC Batch Execution (DIRAC Supercomputer / SLURM)
Submit the full pipeline job to the SLURM workload manager:
```bash
sbatch slurm/submit_pipeline.sh
```

---

## 📊 Outputs & Artifacts

To prevent high disk consumption, intermediate `.mat` files (`standardized_grids.mat`, `basin_time_series.mat`, `grace_reconstructed.mat`, `attribution_results.mat`) are processed in memory and NOT saved to disk.

Only the single **final master result** file and CSV report are saved:
- `outputs/tables/validation_and_trends.mat`: Final master results containing continuous reconstructed TWS series, twin model metrics ($\Delta R^2$, NSE, KGE, RMSE), feature importance, block CV predictions, and modified Mann-Kendall trend statistics.
- `outputs/tables/basin_summary_table.csv`: Exported publication-grade CSV summary table.
