# GEMINI.md - Agent Core Context & Execution Directives

## Project Overview
- **Objective**: Identify global Terrestrial Water Storage ($TWS$) decline trends across the 103 largest river basins and attribute drivers to natural hydroclimate variability vs. human interventions.
- **Governing Mass Balance**:
  $$\frac{dTWS}{dt} = TWSC = P - ET - Q - (GW_{abs} + SW_{abs})$$
- **Input Datasets** ($0.5^\circ \times 0.5^\circ$ resolution, $-180^\circ$ to $180^\circ$, $-90^\circ$ to $90^\circ$, monthly timestep):
  - GRACE/GRACE-FO Ensemble Mean (JPL, GFZ, CSR) $\rightarrow TWS$ anomalies
  - ERA5 $\rightarrow$ Precipitation ($P$, `tp`) and Discharge ($Q$, `dis`)
  - GLEAM $\rightarrow$ Evapotranspiration ($ET$, `E`)
  - PCR-GLOBWB $\rightarrow$ Groundwater Abstraction ($GW_{abs}$) & Surface Water Abstraction ($SW_{abs}$)

## Environment & Infrastructure
- **IDE**: Antigravity IDE
- **HPC Cluster**: DIRAC Supercomputer (IISER Kolkata), SLURM Workload Manager
- **Core Languages**: MATLAB (R2021b or newer) and Bash / SLURM scripts

---

## Agent Behavioral & Coding Directives

### 1. Data Preprocessing & Unit Conversions
- **Mandatory Flux Alignment**: Convert all flux variables ($P, ET, Q, GW_{abs}, SW_{abs}$) to liquid water height equivalent depth (**cm/month**) before running basin spatial aggregation or RF models.
  - ERA5 $P$: Convert meters/month to $cm/month$ ($m \times 100$).
  - GLEAM $ET$: Convert $mm/month$ or $mm/day$ to $cm/month$.
  - ERA5 $Q$: Convert $m^3/s$ to total monthly volume, divide by grid cell/basin spatial area ($m^2$), and convert to $cm/month$.
  - PCR-GLOBWB $GW_{abs}, SW_{abs}$: Convert to $cm/month$.
- **NaN Handling**: Flag and replace all non-physical fill values (`-999`, values $> 1\times 10^{19}$) with MATLAB standard `NaN` immediately upon reading NetCDF variables.
- **Area-Weighted Aggregation**: Always apply latitude cosine weighting (`cosd(lat)`) across the $0.5^\circ$ grid when calculating basin-average time series.

### 2. Memory & Performance Directives (DIRAC HPC)
- **Garbage Collection**: Heavy 3D spatio-temporal NetCDF arrays ($720 \times 360 \times N_{time}$) must be cleared (`clear var_name`) in MATLAB immediately after aggregating data into basin time-series matrices ($N_{time} \times 103$).
- **Parallel Computing**: Vectorize and enforce `parfor` parallel execution across the 103 basins for gap-filling, model tuning, and attribution runs.
- **Headless Execution**: Ensure MATLAB code executes cleanly without GUI calls (`-nodisplay -nosplash -nodesktop`).

### 3. Machine Learning Modeling Standards
- **GRACE Gap-Filling**: Predict missing GRACE/GRACE-FO months (including the 2017–2018 observational gap) using a Random Forest regressor trained on hydroclimate drivers ($P, ET, Q, \text{Temperature}, ONI$) during overlapping continuous observation periods (2002–2017).
- **$TWSC$ Finite Difference**: Calculate $TWSC$ using centered finite differences:
  $$TWSC(t) = \frac{TWS(t+1) - TWS(t-1)}{2 \Delta t}$$
- **Twin Attribution Formulation**:
  - *Model 1 (Natural Baseline $M_{nat}$)*: Predict $TWSC$ using $P, ET, Q$.
  - *Model 2 (Full Anthropogenic $M_{anthro}$)*: Predict $TWSC$ using $P, ET, Q, GW_{abs}, SW_{abs}$.
  - *Attribution Driver Metric*: Compute variance explained gain ($\Delta R^2 = R^2_{anthro} - R^2_{nat}$) alongside feature importance metrics (SHAP values or Out-of-Bag Permutation Importance).

### 4. Rigorous Validation Protocol (Publication Quality)
- **Cross-Validation**: NEVER use standard random K-fold cross-validation (prevents temporal autocorrelation leakage). Use **3-Year Contiguous Block Cross-Validation**.
- **Hydrologic Metrics**: Evaluate model predictions using Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), RMSE, and Pearson $R^2$.
- **Trend Significance**: Calculate trend magnitudes using Theil-Sen's Slope Estimator ($cm/yr$) and assess statistical significance ($p < 0.05$) using the **Hamed and Rao Modified Mann-Kendall Test** (autocorrelation-corrected).

---

## Workspace Directory Architecture
Maintain strict compliance with the project directory structure: