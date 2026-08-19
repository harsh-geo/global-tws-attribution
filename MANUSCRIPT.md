# Disentangling Natural and Anthropogenic Drivers of Global Terrestrial Water Storage Decline using Machine Learning

**Author Name(s)**  
*Affiliation(s), Department, University/Institution*  
**Corresponding Author:** `email@institution.edu`

---

## Abstract

Terrestrial Water Storage (TWS), encompassing surface water, soil moisture, groundwater, and snow/ice reserves, is declining at unprecedented rates across numerous global river basins, posing severe threats to water security, food production, and ecosystem sustainability. While satellite gravimetry missions—specifically the Gravity Recovery and Climate Experiment (GRACE, 2002–2017) and its successor GRACE Follow-On (GRACE-FO, 2018–present)—have revolutionized our ability to monitor large-scale water mass redistribution, disentangling the relative contributions of natural hydroclimate variability from anthropogenic water abstractions remains a formidable challenge. 

Here, we present a Twin Machine Learning Attribution Framework based on Random Forest ensemble regressors, applied across the world’s 103 largest river basins at 0.5° × 0.5° spatial resolution. We first reconstruct the ~11-month observational gap between GRACE and GRACE-FO (July 2017–May 2018) using hydroclimate predictor variables (precipitation, evapotranspiration, runoff, and 2m air temperature) from ERA5 reanalysis and GLEAM datasets. We then quantify long-term TWS trends using the robust Theil-Sen slope estimator and assess their statistical significance via the autocorrelation-corrected Hamed and Rao Modified Mann-Kendall test. 

Our twin attribution models—a Natural Baseline Model ($M_{nat}$) driven solely by climate fluxes and a Full Anthropogenic Model ($M_{anthro}$) augmented with groundwater and surface water abstraction rates from PCR-GLOBWB—enable quantification of the Variance Explained Gain ($\Delta R^2$) attributable to human water abstractions. Results reveal that [XX]% of the 103 basins exhibit statistically significant TWS decline trends. In heavily stressed basins such as the Indus, Tigris-Euphrates, and [other basins], groundwater abstraction emerges as the dominant driver, with $\Delta R^2$ exceeding [XX]. These findings provide actionable, basin-specific attribution diagnostics to inform targeted water resource management and climate adaptation strategies.

**Keywords:** Terrestrial Water Storage; GRACE/GRACE-FO; Machine Learning; Random Forest; Driver Attribution; Groundwater Depletion; Theil-Sen Slope; Modified Mann-Kendall Test

---

## 1. Introduction

Freshwater availability is among the most critical determinants of human well-being, agricultural productivity, and ecosystem health. Terrestrial Water Storage (TWS)—the vertically integrated sum of all water stored on and beneath the Earth’s land surface, including surface water reservoirs, soil moisture, groundwater aquifers, and snow/ice—represents the ultimate buffer against hydroclimatic extremes. Over the past two decades, multiple lines of evidence have converged to reveal an alarming global pattern: TWS is declining in many of the world’s most productive and densely populated river basins (Rodell et al., 2018; Scanlon et al., 2018; Famiglietti, 2014).

The Gravity Recovery and Climate Experiment (GRACE) mission, launched in March 2002, and its successor GRACE Follow-On (GRACE-FO), operational since June 2018, have provided an unprecedented observational record of monthly variations in Earth’s gravity field, from which changes in TWS can be inferred at spatial scales of ~300 km and larger (Tapley et al., 2004; Landerer and Swenson, 2012). These satellite gravimetry observations have unequivocally demonstrated massive water depletions in regions including the Indo-Gangetic Plain (Tiwari et al., 2009), the Central Valley of California (Famiglietti et al., 2011), the Middle East (Voss et al., 2013), and the North China Plain (Feng et al., 2013). The GRACE record, however, contains an approximately 11-month observational gap between the end of the original GRACE mission (June 2017) and the start of GRACE-FO (June 2018), which must be bridged for continuous trend analysis.

A central and unresolved challenge in the interpretation of TWS variability lies in the attribution of observed changes to their underlying drivers. TWS dynamics are governed by the terrestrial water balance:

$$\frac{dTWS}{dt} = TWSC = P - ET - Q - (GW_{abs} + SW_{abs}) \tag{1}$$

where $TWSC$ denotes the Terrestrial Water Storage Change, $P$ is precipitation, $ET$ is evapotranspiration, $Q$ is runoff/discharge, and $GW_{abs}$ and $SW_{abs}$ represent groundwater and surface water abstraction rates, respectively. Observed TWS declines may thus reflect long-term precipitation deficits (e.g., megadroughts exacerbated by climate change), enhanced evapotranspiration driven by rising temperatures, or unsustainable human water withdrawals for irrigation, industrial use, and municipal supply. In practice, these drivers often operate simultaneously and interact nonlinearly, making their separation particularly challenging when relying on simple correlation or regression approaches.

Previous studies have employed hydrological models (e.g., Döll et al., 2014), data assimilation frameworks (e.g., Zaitchik et al., 2008), and statistical decomposition methods (e.g., Humphrey et al., 2017) to attribute TWS variability. However, most approaches suffer from one or more limitations: reliance on a single hydrological model with inherent structural uncertainties, inability to separate climate and anthropogenic signals at the basin scale, or lack of a rigorous statistical framework for trend significance testing that accounts for temporal autocorrelation in hydrological time series.

In this study, we address these gaps through a comprehensive, data-driven attribution framework comprising three methodological pillars:

1. **Machine learning gap-filling**: We reconstruct the GRACE–GRACE-FO observational gap using Random Forest regressors trained on coincident hydroclimate predictor variables, yielding a continuous TWS record spanning April 2002 to December 2019 across 103 major global river basins.
2. **Robust trend estimation**: We quantify long-term TWS trends using the non-parametric Theil-Sen slope estimator and assess their statistical significance using the Hamed and Rao (1998) Modified Mann-Kendall test, which explicitly corrects for temporal autocorrelation—a critical but often neglected consideration in hydrological trend analysis.
3. **Twin machine learning attribution**: We deploy paired Random Forest models—a Natural Baseline Model ($M_{nat}$) driven exclusively by climate variables ($P, ET, Q$) and a Full Anthropogenic Model ($M_{anthro}$) augmented with groundwater and surface water abstraction rates—to isolate the explanatory power gained by including human water use variables, quantified through the Variance Explained Gain ($\Delta R^2$) and Out-of-Bag permutation feature importance.

---

## 2. Data and Methods

### 2.1. Study Domain

The analysis encompasses the world’s 103 largest river basins, which collectively drain approximately 75% of the global continental land area and support the majority of the world’s population. Basin boundaries are delineated using a high-resolution basin mask at 0.5° × 0.5° spatial resolution (720 × 360 grid cells), consistent with the native resolution of the input datasets. All spatial aggregations employ latitude-cosine area weighting ($\cos(\text{lat})$) to account for the decrease in grid cell area toward the poles, ensuring physically meaningful basin-average estimates.

> **[INSERT Figure 1 HERE]**  
> *Description:* A global map (Robinson or Mollweide projection) displaying all 103 river basin polygons, color-coded by basin ID or grouped by continent. Include latitude/longitude grid lines, a scale bar, and labels for the key basins discussed in the paper.  
> **Figure 1.** Study domain showing the 103 largest global river basins used in this analysis. Basins are shaded by unique identifiers overlaid on a global land mask at 0.5° × 0.5° resolution. Major basins discussed in the text (Indus, Tigris-Euphrates, Colorado, Amazon, Congo) are labeled.

---

### 2.2. Datasets

#### 2.2.1. GRACE/GRACE-FO Terrestrial Water Storage Anomalies
Monthly TWS anomaly fields are derived from the ensemble mean of three independent GRACE/GRACE-FO processing centers: the Jet Propulsion Laboratory (JPL), GeoForschungsZentrum (GFZ), and the Center for Space Research (CSR). TWS anomalies are expressed as Liquid Water Equivalent (LWE) thickness in centimeters relative to a 2004–2009 baseline period. The native GRACE temporal coverage extends from April 2002 through June 2017, with GRACE-FO resuming in June 2018. Missing months within the operational periods (instrument anomalies, orbit maneuvers) and the ~11-month inter-mission gap (July 2017–May 2018) are explicitly represented as NaN values in the analysis timeline, pending reconstruction (Section 2.4).

#### 2.2.2. ERA5 Reanalysis: Precipitation, Runoff, and 2m Air Temperature
Monthly total precipitation ($P$), total runoff ($Q$), and 2m air temperature ($T$, variable name `t2m`) fields are obtained from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 reanalysis product (Hersbach et al., 2020). ERA5 provides globally complete, hourly atmospheric fields at 0.25° resolution, aggregated here to monthly totals (or monthly means for $T$) on a 0.5° × 0.5° grid. Precipitation is converted from meters/month to centimeters/month ($\times 100$), and runoff is similarly converted from meters/month to centimeters/month to maintain dimensional consistency across all water balance terms. Monthly mean 2m air temperature is retained in Kelvin and used exclusively as an auxiliary predictor for the GRACE gap-filling model (Section 2.4); it does not enter the water balance equation directly.

#### 2.2.3. GLEAM Evapotranspiration
Actual evapotranspiration ($ET$) estimates are sourced from the Global Land Evaporation Amsterdam Model (GLEAM) v3.x (Martens et al., 2017; Miralles et al., 2011). GLEAM provides satellite-observation-driven ET estimates at 0.25° resolution, regridded to 0.5° × 0.5° for consistency. The native GLEAM units of millimeters/month are converted to centimeters/month ($\div 10$). Spatial alignment requires a latitude flip (North$\to$South to South$\to$North) and a 180° longitudinal shift to match the $-180^\circ$ to $+180^\circ$ convention used throughout this analysis.

#### 2.2.4. PCR-GLOBWB Groundwater and Surface Water Abstraction
Spatially explicit estimates of groundwater abstraction ($GW_{abs}$) and surface water abstraction ($SW_{abs}$) are obtained from the PCR-GLOBWB 2.0 global hydrological model (Sutanudjaja et al., 2018). PCR-GLOBWB simulates global water demand, allocation, and abstraction from both groundwater and surface water sources at 0.5° resolution, accounting for irrigation, domestic, and industrial water use sectors. Abstraction rates are converted from meters/month to centimeters/month ($\times 100$).

> **[INSERT Table 1 HERE]**  
> *Description:* A multi-column table with rows for each dataset: GRACE/GRACE-FO TWS, ERA5 P, ERA5 Q, ERA5 T2m, GLEAM ET, PCR-GLOBWB GW_abs, PCR-GLOBWB SW_abs. Columns: Variable, Source, Native Resolution, Unit Conversion, Temporal Coverage, Reference.  
> **Table 1.** Summary of input datasets used in this study, including variable names, native units, converted units (cm/month or K for temperature), spatial resolution, temporal coverage, and source references.

---

### 2.3. Data Preprocessing and Unit Harmonization

All gridded datasets are preprocessed to ensure spatial, temporal, and dimensional consistency prior to analysis. The preprocessing pipeline (implemented in MATLAB R2021b) proceeds as follows:

1. **Non-physical fill values:** All NetCDF variables are scanned for non-physical fill values ($\le -900$, $> 1 \times 10^{19}$, $\pm\text{Inf}$) and replaced with MATLAB `NaN` immediately upon ingestion, preventing contamination of downstream calculations.
2. **Unit harmonization:** All hydroclimate fluxes ($P, ET, Q, GW_{abs}, SW_{abs}$) and TWS anomalies are converted to a common unit of liquid water height equivalent depth (cm/month) prior to any spatial aggregation or modeling.
3. **Temporal alignment:** All datasets are re-indexed onto a common 213-month timeline (April 2002 through December 2019). GRACE months that are observationally present are mapped using their native timestamps; missing months are encoded as `NaN` for subsequent gap-filling.
4. **Area-weighted basin aggregation:** Gridded fields are spatially aggregated over each of the 103 river basins using latitude-cosine weighting. For each basin $b$ and variable $X$, the basin-average time series is computed as:

$$\bar{X}_b(t) = \frac{\sum_i [X(i, t) \times \cos(\text{lat}_i)]}{\sum_i [\cos(\text{lat}_i)]} \quad \text{for all grid cells } i \in \text{basin } b \tag{2}$$

where $\text{lat}_i$ is the latitude of grid cell $i$. This yields six basin-average time series matrices of dimension $[213 \times 103]$: $TWS, P, ET, Q, GW_{abs},$ and $SW_{abs}$.

---

### 2.4. Machine Learning Gap-Filling of GRACE Observations

To obtain a continuous TWS record, we reconstruct missing GRACE months—including the ~11-month inter-mission gap (July 2017–May 2018) and sporadic within-mission dropouts—using Random Forest (RF) ensemble regressors (Breiman, 2001). For each basin $b$, a separate RF model is trained on the subset of months where GRACE TWS observations are available and predictor variables ($P, ET, Q, T$, and the water balance residual $P - ET - Q$) are non-missing, where $T$ is the basin-average monthly mean 2m air temperature from ERA5 (`t2m`). Including $T$ as a predictor improves gap-filling skill by capturing thermodynamic controls on evapotranspiration and snowmelt that are not fully represented by the flux variables alone. The trained model is then applied to predict TWS anomalies at all gap months where predictors are available.

RF hyperparameters are set to 200 trees (`n_trees = 200`) with a minimum leaf size of 5 (`min_leaf_size = 5`) to balance predictive accuracy and regularization against overfitting. Model skill is evaluated using Out-of-Bag (OOB) predictions, which provide an unbiased estimate of generalization error without requiring a separate validation set. The OOB Root Mean Square Error (RMSE) and coefficient of determination ($R^2$) are recorded for each basin. All 103 basins are processed in parallel using MATLAB’s `parfor` construct for computational efficiency on the DIRAC HPC cluster.

> **[INSERT Figure 2 HERE]**  
> *Description:* Three-panel figure. Panel (a): scatter plot with color-coded density. Panel (b): time series plot showing observed TWS with markers, continuous reconstructed TWS as a line, and a gray or red shaded region for the 2017–2018 gap. Panel (c): distribution of OOB R² across all 103 basins showing the quality of reconstruction.  
> **Figure 2.** Validation of the Random Forest gap-filling model. (a) Scatter plot of OOB-predicted versus observed TWS anomalies for a representative basin (e.g., Indus or Amazon), with 1:1 reference line, R², and RMSE annotated. (b) Time series of observed GRACE TWS (circles), gap-filled TWS (red line), and the shaded inter-mission gap period (July 2017–May 2018) for the same basin. (c) Histogram or boxplot of OOB R² values across all 103 basins.

---

### 2.5. Computation of TWS Change (TWSC)

The rate of change of TWS ($TWSC$) is computed from the gap-filled, continuous TWS time series using centered finite differences:

$$TWSC(t) = \frac{TWS(t+1) - TWS(t-1)}{2\Delta t} \tag{3}$$

where $\Delta t = 1\text{ month}$. Forward and backward differences are applied at the first and last time steps, respectively. Prior to attribution modeling, all variables ($TWSC, P, ET, Q, GW_{abs}, SW_{abs}$) are deseasonalized by subtracting the long-term monthly climatological mean, removing the dominant seasonal cycle and isolating interannual to decadal anomaly signals.

---

### 2.6. Twin Random Forest Attribution Framework

The core methodological innovation of this study is a twin machine learning framework designed to isolate the explanatory contribution of anthropogenic water abstractions to observed TWS variability. Two Random Forest regression models are trained independently for each basin:

**Model 1—Natural Baseline ($M_{nat}$):** Predicts $TWSC$ as a function of natural hydroclimate drivers alone:
$$TWSC_{pred} = f_{nat}(P, ET, Q) \tag{4}$$

**Model 2—Full Anthropogenic ($M_{anthro}$):** Predicts $TWSC$ as a function of both natural and anthropogenic drivers:
$$TWSC_{pred} = f_{anthro}(P, ET, Q, GW_{abs}, SW_{abs}) \tag{5}$$

Both models use 200 trees with a minimum leaf size of 10 and are evaluated using OOB predictions. The OOB $R^2$ is computed for each model, and the Variance Explained Gain is defined as:

$$\Delta R^2 = R^2_{anthro} - R^2_{nat} \tag{6}$$

A positive $\Delta R^2$ indicates that human water abstractions explain a statistically meaningful fraction of TWS variability beyond what can be attributed to natural climate forcing. To identify the single most important driver for each basin, we extract Out-of-Bag Permutation Feature Importance scores from $M_{anthro}$, which quantify the increase in OOB prediction error when each feature’s values are randomly permuted, thereby breaking its relationship with the target variable.

> **[INSERT Figure 3 HERE]**  
> *Description:* A conceptual/methodological flowchart showing the twin model architecture. Input boxes for each predictor variable, two parallel Random Forest model boxes, output arrows to TWSC predictions, and the ΔR² comparison step. Use a clean, schematic style suitable for publication.  
> **Figure 3.** Schematic diagram of the Twin Random Forest Attribution Framework. Left branch: $M_{nat}$ (Natural Baseline) trained on $P, ET, Q$. Right branch: $M_{anthro}$ (Full Anthropogenic) trained on $P, ET, Q, GW_{abs}, SW_{abs}$. The Variance Explained Gain ($\Delta R^2$) is computed as the difference in OOB $R^2$ between the two models.

---

### 2.7. Trend Estimation and Significance Testing

Long-term TWS trends are estimated using the Theil-Sen slope estimator (Theil, 1950; Sen, 1968), a non-parametric method that computes the median of all pairwise slopes between data points, providing robustness against outliers and non-normality. The estimated slope $\hat{\beta}$ (in cm/year) represents the rate of TWS change over the analysis period.

The statistical significance of each basin’s trend is assessed using the Modified Mann-Kendall (MK) test of Hamed and Rao (1998), which corrects the variance of the Mann-Kendall $S$ statistic for temporal autocorrelation in the data. The correction factor accounts for significant autocorrelation coefficients (at the 95% confidence level) up to lag $n/4$, where $n$ is the series length. A trend is declared statistically significant at the $\alpha = 0.05$ level. This autocorrelation correction is critical for hydrological time series, where persistence (e.g., multi-year droughts, sustained pumping) can inflate apparent trend significance if standard MK tests are naïvely applied.

---

### 2.8. Model Validation: 3-Year Contiguous Block Cross-Validation

To rigorously evaluate the predictive skill of both $M_{nat}$ and $M_{anthro}$ without temporal information leakage, we employ 3-Year Contiguous Block Cross-Validation (Block CV). Standard random K-fold cross-validation is inappropriate for autocorrelated time series because random partitioning allows temporally adjacent observations to appear in both training and test sets, artificially inflating performance metrics. In Block CV, the 213-month record is divided into non-overlapping contiguous blocks of 36 months (3 years). For each fold, one block is held out as the test set while all remaining blocks serve as the training set. Models are trained *de novo* on the training blocks and evaluated on the held-out test block.

Four hydrologic performance metrics are computed for each basin across all Block CV folds:
1. Nash-Sutcliffe Efficiency (NSE),
2. Kling-Gupta Efficiency (KGE, using a modified formulation suitable for zero-mean anomaly variables),
3. Root Mean Square Error (RMSE, cm/month), and
4. Pearson $R^2$.

These metrics collectively assess correlation, bias, variability ratio, and absolute error magnitude.

---

## 3. Results

### 3.1. Global Patterns of TWS Decline

Application of the Theil-Sen slope estimator to the gap-filled, continuous TWS record (April 2002–December 2019) reveals widespread negative TWS trends across the 103 basins analyzed (Figure 4). [XX] out of 103 basins ([XX]%) exhibit statistically significant ($p < 0.05$, Modified Mann-Kendall test) negative trends, with a mean decline rate of [XX] cm/year among these basins. The most severe depletions are observed in [basin names], with trend magnitudes exceeding [XX] cm/year. Conversely, [XX] basins show significant positive TWS trends, primarily located in [regions], likely reflecting increased precipitation or glacier/snowmelt contributions.

> **[INSERT Figure 4 HERE]**  
> *Description:* A global choropleth map with each basin polygon colored by its Theil-Sen TWS trend slope. Use a diverging Red-White-Blue colormap symmetric about zero. Overlay stipple dots on non-significant basins. Include a colorbar labeled 'TWS Trend (cm/year)' and a title. Generated by `plot_basin_trends.m`.  
> **Figure 4.** Global map of TWS trends (cm/year) across the 103 largest river basins for the period April 2002–December 2019. A diverging blue (positive/gaining)–red (negative/declining) color scale centered at zero is used. Basins with statistically non-significant trends ($p \ge 0.05$, Hamed–Rao Modified Mann-Kendall test) are overlaid with gray stippling. Non-basin areas are shown in light gray.

> **[INSERT Figure 5 HERE]**  
> *Description:* A horizontal or vertical bar chart showing the top-20 declining basins with their Theil-Sen trend slopes. Basin names or IDs on the y-axis. Significance indicated by bar color.  
> **Figure 5.** Bar chart of the 20 basins with the most negative TWS trends (cm/year), ranked from most severe to least severe. Bars are colored red for statistically significant trends and gray for non-significant trends. Error bars or confidence intervals from the Theil-Sen estimator may be included.

---

### 3.2. Validation of Machine Learning Models

The 3-Year Contiguous Block Cross-Validation demonstrates that both twin attribution models achieve robust predictive skill across the majority of basins. The Natural Baseline Model ($M_{nat}$) achieves a median Block CV NSE of [XX] (interquartile range: [XX–XX]), median KGE of [XX], and median RMSE of [XX] cm/month. The Full Anthropogenic Model ($M_{anthro}$) achieves improved performance with a median NSE of [XX], median KGE of [XX], and median RMSE of [XX] cm/month (Table 2). These results confirm that the models capture the dominant hydrological dynamics without overfitting, as the Block CV procedure explicitly prevents temporal leakage.

> **[INSERT Table 2 HERE]**  
> *Description:* A table with rows: NSE, KGE, RMSE (cm/month), R². Columns grouped by $M_{nat}$ and $M_{anthro}$, each showing Median, Mean, IQR, and % basins with NSE > 0.  
> **Table 2.** Summary of 3-Year Block Cross-Validation performance metrics for the Natural Baseline ($M_{nat}$) and Full Anthropogenic ($M_{anthro}$) models across all 103 basins. Metrics include median, mean, interquartile range, and percentage of basins with NSE > 0 (skillful).

> **[INSERT Figure 6 HERE]**  
> *Description:* Side-by-side boxplots for each metric. Two boxes per metric ($M_{nat}$ and $M_{anthro}$). NSE and KGE should show that $M_{anthro}$ generally matches or exceeds $M_{nat}$. RMSE should show that $M_{anthro}$ generally has equal or lower error.  
> **Figure 6.** Box-and-whisker plots comparing Block Cross-Validation performance metrics (NSE, KGE, RMSE) between the Natural Baseline ($M_{nat}$, blue) and Full Anthropogenic ($M_{anthro}$, orange) models across all 103 basins. Whiskers extend to the 5th and 95th percentiles; outliers are plotted as individual points.

---

### 3.3. Attribution: Disentangling Natural and Anthropogenic Drivers

The twin attribution framework reveals heterogeneous driver dominance across the global basin network. Figure 7 presents a global map of the dominant driver for each basin, defined as the feature with the highest OOB permutation importance in $M_{anthro}$. Precipitation ($P$) emerges as the dominant driver in [XX]% of basins, predominantly in tropical and high-latitude regions. Evapotranspiration ($ET$) dominates in [XX]% of basins, particularly in semi-arid and warming regions. Groundwater abstraction ($GW_{abs}$) is identified as the dominant driver in [XX]% of basins, concentrated in South Asia (Indus, Ganges), the Middle East (Tigris-Euphrates), and parts of North America and North Africa.

> **[INSERT Figure 7 HERE]**  
> *Description:* A global choropleth map with each basin polygon colored by its dominant driver (5 categories). Use the colormap: P=Blue, ET=Green, Q=Cyan, GW=Orange, SW=Red. Include a categorical colorbar/legend. Generated by `plot_global_attribution_map.m`.  
> **Figure 7.** Global map of the dominant driver of TWS variability in each of the 103 basins, determined by maximum OOB permutation feature importance from $M_{anthro}$. Basins are color-coded by dominant driver category: Precipitation (blue), Evapotranspiration (green), Runoff (cyan), Groundwater Abstraction (orange), Surface Water Abstraction (red). A categorical legend is provided.

The Variance Explained Gain ($\Delta R^2$) provides a quantitative measure of the additional explanatory power contributed by anthropogenic abstractions. Across all 103 basins, the mean $\Delta R^2$ is [XX] ($\pm$ [XX] standard deviation). However, the distribution is highly skewed: basins with known intensive groundwater exploitation exhibit $\Delta R^2$ values exceeding [XX], while basins with minimal human abstraction show near-zero or slightly negative $\Delta R^2$ values (Figure 8).

> **[INSERT Figure 8 HERE]**  
> *Description:* A sorted bar chart or lollipop plot with basin ID/name on the x-axis and ΔR² on the y-axis. Bars colored by dominant driver. Label the top 10 and bottom 5 basins.  
> **Figure 8.** Scatter plot or bar chart of Variance Explained Gain ($\Delta R^2 = R^2_{anthro} - R^2_{nat}$) for the 103 basins, sorted from highest to lowest $\Delta R^2$. Basins are colored by their dominant driver category. Key basins (Indus, Tigris-Euphrates, Colorado, etc.) are labeled. A dashed horizontal line at $\Delta R^2 = 0$ separates basins where anthropogenic variables improve model performance from those where they do not.

> **[INSERT Figure 9 HERE]**  
> *Description:* A horizontal stacked bar chart for the 15 most severely declining basins. Each bar is segmented into 5 features (P, ET, Q, GW_abs, SW_abs) with consistent colors matching Figure 7. Annotate where GW_abs or SW_abs dominate.  
> **Figure 9.** Stacked or grouped bar chart of OOB permutation feature importance for the top 15 most stressed basins (highest negative TWS trend $\times$ significant). Five stacked segments per bar represent $P, ET, Q, GW_{abs}$, and $SW_{abs}$ importance. Basin names are shown on the y-axis.

---

## 4. Discussion

### 4.1. Hotspots of Human-Induced TWS Depletion

Our twin attribution framework identifies several well-documented hotspots of anthropogenic TWS depletion, lending credibility to the methodology and providing new quantitative attribution metrics.

- **Indus Basin:** The Indus Basin, encompassing the intensively irrigated Indo-Gangetic aquifer system, exhibits one of the most severe TWS decline rates globally ([XX] cm/year, $p < 0.001$). The $\Delta R^2$ for this basin is [XX], indicating that groundwater abstraction explains [XX]% of additional TWS variance beyond natural climate drivers. OOB feature importance confirms $GW_{abs}$ as the dominant driver (importance = [XX]), consistent with extensive literature documenting unsustainable groundwater extraction for rice-wheat cropping systems (Tiwari et al., 2009; Rodell et al., 2009).
- **Tigris-Euphrates Basin:** The Tigris-Euphrates system, spanning Turkey, Syria, and Iraq, shows a significant negative TWS trend of [XX] cm/year. The attribution analysis reveals a $\Delta R^2$ of [XX], with both $GW_{abs}$ and $SW_{abs}$ contributing substantially to the observed depletion. This is consistent with the combined effects of upstream dam construction, expanding irrigation networks, and drought conditions reported by Voss et al. (2013) and Joodaki et al. (2014).
- **Colorado Basin:** The Colorado River Basin shows [describe findings]. The $\Delta R^2$ of [XX] suggests [interpretation]. [Discuss in relation to Castle et al. (2014) findings on Lake Mead and groundwater depletion.]

> **[INSERT Figure 10 HERE]**  
> *Description:* A 3×3 panel figure. Three columns for three basins. Row 1: TWS time series with observed (markers) and reconstructed (line). Row 2: TWSC comparison — observed vs. M_nat prediction vs. M_anthro prediction. Row 3: horizontal bar chart of feature importance (5 features) for each basin.  
> **Figure 10.** Detailed case study panels for three key basins: (a) Indus, (b) Tigris-Euphrates, (c) Colorado. Each panel shows: (top) time series of observed and reconstructed TWS anomalies; (middle) observed TWSC versus $M_{nat}$ and $M_{anthro}$ predictions; (bottom) feature importance bar chart for $M_{anthro}$. Basin boundaries are shown in an inset map.

---

### 4.2. Climate-Driven TWS Variability

Not all TWS declines are attributable to direct human water extraction. Our analysis identifies a substantial subset of basins where precipitation deficits and/or enhanced evapotranspiration are the primary drivers of TWS loss. In these basins, the $\Delta R^2$ is near zero or slightly negative, indicating that the addition of anthropogenic variables provides no improvement in predictive skill.

For example, [basin names in drought-prone regions] show significant TWS decline trends driven predominantly by multi-year precipitation deficits associated with large-scale climate modes such as the El Niño–Southern Oscillation (ENSO) and the Pacific Decadal Oscillation (PDO). In [other basins], increasing evapotranspiration driven by rising air temperatures appears to be the dominant mechanism, consistent with projections of intensified atmospheric water demand under continued warming (Jung et al., 2010; Zhang et al., 2016).

---

### 4.3. Uncertainties and Limitations

Several sources of uncertainty affect the results of this study:

1. **GRACE Resolution & Spatial Leakage:** The spatial resolution of GRACE (~300 km effective resolution) limits the ability to resolve sub-basin heterogeneity in TWS changes, particularly in smaller basins where signal leakage from neighboring regions may influence basin-average estimates.
2. **Reanalysis & Flux Product Biases:** The hydroclimate predictor datasets (ERA5 reanalysis, GLEAM ET) are themselves model products with inherent biases and uncertainties, which propagate into the Random Forest predictions.
3. **Abstraction Data Incompleteness:** The PCR-GLOBWB abstraction estimates are based on national and sub-national water use statistics that may be outdated or incomplete, particularly in data-sparse regions.
4. **Machine Learning Model Structure:** The Random Forest framework, while powerful in capturing nonlinear relationships, is not a physically constrained model and may conflate correlated drivers. The OOB permutation importance metric assumes feature independence, which is violated when predictors are correlated (e.g., $P$ and $Q$). Future work should explore conditional permutation importance (Strobl et al., 2008) or SHAP (SHapley Additive exPlanations) values to provide more robust and interpretable feature attribution.
5. **Decadal Temporal Length:** Finally, the 213-month analysis period (2002–2019), while representing the longest available satellite gravimetry record, may be insufficient to fully characterize decadal-scale climate variability modes, potentially aliasing multi-decadal signals as secular trends.

---

## 5. Conclusions

This study presents a comprehensive, data-driven framework for identifying and attributing global Terrestrial Water Storage (TWS) decline trends across the world’s 103 largest river basins. Our principal conclusions are:

1. **Widespread TWS decline:** [XX] out of 103 basins ([XX]%) exhibit statistically significant negative TWS trends over the GRACE/GRACE-FO era (2002–2019), with a mean decline rate of [XX] cm/year among these basins.
2. **Effective gap reconstruction:** Random Forest-based gap-filling successfully reconstructs the GRACE–GRACE-FO observational gap with a mean Out-of-Bag $R^2$ of [XX] across all basins, enabling continuous trend analysis.
3. **Anthropogenic attribution:** The Twin Random Forest Attribution Framework demonstrates that groundwater and surface water abstractions provide significant additional explanatory power ($\Delta R^2 > \text{[XX]}$) in [XX] basins, predominantly in South Asia, the Middle East, and [other regions], identifying these as hotspots of human-induced water depletion.
4. **Climate-driven variability:** In the remaining basins, TWS variability is predominantly explained by natural hydroclimate drivers—principally precipitation deficits and enhanced evapotranspiration—highlighting the combined impacts of climate variability and global warming on freshwater reserves.
5. **Policy implications:** The basin-specific attribution diagnostics produced by this framework can directly inform targeted water resource management strategies, distinguishing basins where demand-side interventions (regulation of groundwater pumping, improved irrigation efficiency) are most urgently needed from those where climate adaptation measures (drought preparedness, reservoir management) are the priority.

Future extensions of this work will incorporate SHAP-based feature attribution for improved interpretability, expand the analysis to sub-basin scales where data resolution permits, and extend the temporal coverage as the GRACE-FO record lengthens.

---

## Acknowledgments

[The authors acknowledge computational resources provided by the DIRAC Supercomputer at IISER Kolkata. GRACE/GRACE-FO data were obtained from [source]. ERA5 data were provided by ECMWF through the Copernicus Climate Data Store. GLEAM data were obtained from [source]. PCR-GLOBWB simulations were provided by [source]. This work was supported by [funding agency/grant number].]

---

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Castle, S. L., Thomas, B. F., Reager, J. T., Rodell, M., Swenson, S. C., & Famiglietti, J. S. (2014). Groundwater depletion during drought threatens future water security of the Colorado River Basin. *Geophysical Research Letters*, 41(16), 5904–5911.
- Döll, P., Müller Schmied, H., Schuh, C., Portmann, F. T., & Eicker, A. (2014). Global-scale assessment of groundwater depletion and related groundwater abstractions: Combining hydrological modeling with information from well observations and GRACE satellites. *Water Resources Research*, 50(7), 5698–5720.
- Famiglietti, J. S. (2014). The global groundwater crisis. *Nature Climate Change*, 4(11), 945–948.
- Famiglietti, J. S., Lo, M., Ho, S. L., Bethune, J., Anderson, K. J., Syed, T. H., Swenson, S. C., de Linage, C. R., & Rodell, M. (2011). Satellites measure recent rates of groundwater depletion in California’s Central Valley. *Geophysical Research Letters*, 38(3), L03403.
- Feng, W., Zhong, M., Lemoine, J. M., Biancale, R., Hsu, H. T., & Xia, J. (2013). Evaluation of groundwater depletion in North China using the Gravity Recovery and Climate Experiment (GRACE) data and ground-based measurements. *Water Resources Research*, 49(4), 2110–2118.
- Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204(1–4), 182–196.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
- Humphrey, V., Gudmundsson, L., & Seneviratne, S. I. (2017). A global reconstruction of climate-driven subdecadal water storage variability. *Geophysical Research Letters*, 44(5), 2300–2309.
- Joodaki, G., Wahr, J., & Swenson, S. (2014). Estimating the human contribution to groundwater depletion in the Middle East, from GRACE data, land surface models, and well observations. *Water Resources Research*, 50(3), 2679–2692.
- Jung, M., et al. (2010). Recent decline in the global land evapotranspiration trend due to limited moisture supply. *Nature*, 467(7318), 951–954.
- Landerer, F. W., & Swenson, S. C. (2012). Accuracy of scaled GRACE terrestrial water storage estimates. *Water Resources Research*, 48(4), W04531.
- Martens, B., et al. (2017). GLEAM v3: satellite-based land evaporation and root-zone soil moisture. *Geoscientific Model Development*, 10(5), 1903–1925.
- Miralles, D. G., Holmes, T. R. H., De Jeu, R. A. M., Gash, J. H., Meesters, A. G. C. A., & Dolman, A. J. (2011). Global land-surface evaporation estimated from satellite-based observations. *Hydrology and Earth System Sciences*, 15(2), 453–469.
- Rodell, M., Velicogna, I., & Famiglietti, J. S. (2009). Satellite-based estimates of groundwater depletion in India. *Nature*, 460(7258), 999–1002.
- Rodell, M., et al. (2018). Emerging trends in global freshwater availability. *Nature*, 557(7707), 651–659.
- Scanlon, B. R., et al. (2018). Global models underestimate large decadal declining and rising water storage trends relative to GRACE satellite data. *Proceedings of the National Academy of Sciences*, 115(6), E1080–E1089.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall’s tau. *Journal of the American Statistical Association*, 63(324), 1379–1389.
- Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics*, 9, 307.
- Sutanudjaja, E. H., et al. (2018). PCR-GLOBWB 2: a 5 arcmin global hydrological and water resources model. *Geoscientific Model Development*, 11(6), 2429–2453.
- Tapley, B. D., Bettadpur, S., Ries, J. C., Thompson, P. F., & Watkins, M. M. (2004). GRACE measurements of mass variability in the Earth system. *Science*, 305(5683), 503–505.
- Theil, H. (1950). A rank-invariant method of linear and polynomial regression analysis. *Proceedings of the Royal Netherlands Academy of Arts and Sciences*, 53, 386–392, 521–525, 1397–1412.
- Tiwari, V. M., Wahr, J., & Swenson, S. (2009). Dwindling groundwater resources in northern India, from satellite gravity observations. *Geophysical Research Letters*, 36(18), L18401.
- Voss, K. A., Famiglietti, J. S., Lo, M., De Linage, C., Rodell, M., & Swenson, S. C. (2013). Groundwater depletion in the Middle East from GRACE with implications for transboundary water management in the Tigris-Euphrates-Western Iran region. *Water Resources Research*, 49(2), 904–914.
- Zaitchik, B. F., Rodell, M., & Reichle, R. H. (2008). Assimilation of GRACE terrestrial water storage data into a land surface model: Results for the Mississippi River basin. *Journal of Hydrometeorology*, 9(3), 535–548.
- Zhang, Y., et al. (2016). Multi-decadal trends in global terrestrial evapotranspiration and its components. *Scientific Reports*, 6, 19124.

---

## Supplementary Information (Outline)

The following supplementary materials accompany this manuscript:

- **Table S1:** Complete basin-by-basin summary table including Basin ID, TWS trend (cm/year), Modified Mann-Kendall p-value, significance flag, Block CV NSE ($M_{nat}$ and $M_{anthro}$), Block CV KGE ($M_{nat}$ and $M_{anthro}$), OOB $R^2$ ($M_{nat}$ and $M_{anthro}$), $\Delta R^2$, and dominant driver category for all 103 basins.
- **Figure S1:** Individual basin time series of observed and reconstructed TWS anomalies for all 103 basins (multi-page panel figure).
- **Figure S2:** Global map of OOB $R^2$ for the gap-filling Random Forest model, showing spatial variability in reconstruction quality.
- **Figure S3:** Correlation matrix of predictor variables ($P, ET, Q, GW_{abs}, SW_{abs}$) across representative basins, illustrating inter-predictor dependencies.
- **Code and Data Availability:** All MATLAB code used in this study is available at `[repository URL]`. Processed basin-level time series and attribution results are archived at `[DOI/repository]`.
