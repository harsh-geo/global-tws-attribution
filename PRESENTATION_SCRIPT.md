# Speech Script — 10-Minute Talk

**Title:** *Disentangling Natural and Anthropogenic Drivers of Global Terrestrial Water Storage Decline using Machine Learning*

**Duration:** ~10 minutes (~1500 words)

**Visual aids referenced:** Equations, tables, and figures are indicated inline as `[EQUATION]`, `[TABLE]`, and `[FIGURE]` blocks. Figures already generated are marked with their filenames.

---

Good morning everyone. Today I want to talk about a question that sits at the intersection of hydrology, remote sensing, and machine learning — *why is the world losing freshwater, and can we tell apart the role of climate from the role of human activity?*

### Beat 1 — What Is Happening: The Global TWS Crisis

Terrestrial Water Storage — TWS — is the total stock of water on and beneath the land surface: lakes, soil moisture, snowpack, and groundwater. It is, in the simplest terms, the freshwater reservoir that 8 billion people, and every agricultural system and ecosystem on the planet, ultimately depend upon. In 2018, Rodell and colleagues published a landmark study in *Nature* showing that freshwater availability is declining in many of the world's most critical basins — the Indus, the Tigris-Euphrates, the Central Valley of California. Scanlon et al., also in 2018, demonstrated in *PNAS* that global hydrological models systematically underestimate these declining trends compared to what the satellites observe.

But identifying that water is disappearing is only half the problem. The harder question is *attribution*: is a given basin losing water because of prolonged drought? Because warming temperatures are increasing evapotranspiration? Or because humans are pumping groundwater faster than it recharges? This distinction is not academic — it directly determines whether the correct policy response is drought preparedness, or pumping regulation, or irrigation reform.

### Beat 2 — What Others Have Tried, and Where They Fall Short

So what has the field done about the attribution problem? Prior work falls into three streams. First, regional case studies: Tiwari et al. documented groundwater depletion in the Indus, Famiglietti et al. in the California Central Valley, Voss et al. in the Tigris-Euphrates. These are excellent basin-specific studies — but they cover individual basins only, not the global picture. Second, physics-based model simulations: Döll et al. ran a global hydrological model with and without human water use modules and compared the outputs to GRACE. Zaitchik et al. assimilated GRACE data directly into a land surface model. Humphrey et al. used a statistical decomposition approach. But these approaches all share a fundamental limitation: they depend on the structural assumptions of a single hydrological model. If the model's parameterisation of infiltration, aquifer properties, or irrigation schemes is wrong, the attribution is wrong — and no independent data-driven check is provided.

Let me be more specific about the gaps. There are five. **Gap 1**: single-model structural uncertainty — the attribution inherits whatever biases the chosen model has. **Gap 2**: no global-scale comparison — no one has performed systematic basin-by-basin driver separation across *all* major global basins simultaneously, which prevents cross-regime assessment. **Gap 3**: temporal autocorrelation leakage — many ML studies in hydrology use random K-fold cross-validation, which shuffles the time indices, allows adjacent months in both train and test sets, and artificially inflates skill metrics. **Gap 4**: the GRACE observational gap — the ~11-month gap between GRACE and GRACE-FO (July 2017 to May 2018) is typically excluded or linearly interpolated, truncating trend analysis during a critical drought period. **Gap 5**: weak signal masking — when you put both climate variables and abstraction variables into a standard Random Forest, the algorithm overwhelmingly splits on the high-variance climate variables — precipitation swings by 10–20 cm/month, while groundwater abstraction changes by fractions of a centimetre. The anthropogenic signal gets buried.

### Beat 3 — What We Do Differently

Our work addresses all five of these gaps. We developed a *twin machine learning framework* — two Random Forest models, one seeing only climate ($P$, $ET$, $Q$), the other seeing climate plus human abstraction ($GW_{abs}$, $SW_{abs}$). By comparing their predictive power, we isolate the explanatory contribution of human water use — without relying on any single hydrological model's assumptions (Gap 1). We apply this systematically across the world's 103 largest river basins — the first study to do so at this scale (Gap 2). We fill the GRACE gap using a covariate-driven Random Forest reconstruction rather than simple interpolation (Gap 4). We validate using 3-year contiguous block cross-validation with a fixed-baseline deseasonalisation strategy, which prevents both temporal autocorrelation leakage and deseasonalisation leakage (Gap 3). And we employ a forced feature-sampling strategy — setting `NumPredictorsToSample` to 1 — which forces the ensemble to build trees using every predictor, including the low-variance abstraction signals, thereby amplifying weak but physically meaningful human impacts (Gap 5).

### The Observational Record

Our TWS observations come from NASA's GRACE and GRACE-FO satellite missions. These spacecraft measure monthly changes in Earth's gravity field, from which water mass anomalies can be inferred at scales of roughly 300 kilometres and larger — as documented by Tapley et al. in their 2004 *Science* paper. We use the ensemble mean of three independent processing centres — JPL, GFZ, and CSR — to reduce noise, giving us a record spanning April 2002 to December 2019.

There is one gap: GRACE ended in June 2017, and GRACE-FO launched in June 2018, leaving roughly 11 months of missing data. We filled this using a Random Forest regressor trained on coincident precipitation, evapotranspiration, and runoff — achieving a mean out-of-bag $R^2$ across all basins that confirms the reconstruction is reliable.

`[FIGURE: basin_51_timeseries.png — TWS time series for the Ganga basin showing observed GRACE data, reconstructed values, and the shaded gap period]`

`[FIGURE: global_oob_r2_histogram.png — Distribution of OOB R² for the gap-filling model across all 103 basins]`

### The Governing Equation

The physics is grounded in a simple mass balance:

$$\frac{dTWS}{dt} = TWSC = P - ET - Q - (GW_{abs} + SW_{abs})$$

| Symbol | Meaning | Data Source |
|--------|---------|-------------|
| $P$ | Precipitation | ERA5 reanalysis (Hersbach et al., 2020) |
| $ET$ | Evapotranspiration | GLEAM v3.x (Martens et al., 2017) |
| $Q$ | Runoff / discharge | ERA5 reanalysis |
| $GW_{abs}$ | Groundwater abstraction | PCR-GLOBWB 2.0 (Sutanudjaja et al., 2018) |
| $SW_{abs}$ | Surface water abstraction | PCR-GLOBWB 2.0 |

All fluxes are harmonised to centimetres per month of liquid water equivalent, and spatially aggregated over each of the 103 largest global river basins using latitude-cosine area weighting. Before modelling, we remove the seasonal cycle by subtracting the monthly climatological mean computed from a fixed 2004–2009 baseline. The reason for this fixed baseline — rather than computing it from the full time series — is to prevent information from the test period leaking into the training set during cross-validation. This is a subtle but important point that I will return to.

### The Twin Model Framework

Now — the core of this work. We train two Random Forest models independently for each basin.

**Model 1**, the Natural Baseline:
$$TWSC = f_{nat}(P, \ ET, \ Q)$$

This model can only see climate variables. It asks: *how much of the observed water storage change can climate explain by itself?*

**Model 2**, the Full Anthropogenic model:
$$TWSC = f_{anthro}(P, \ ET, \ Q, \ GW_{abs}, \ SW_{abs})$$

This model sees everything — climate plus human abstraction. It asks: *does adding human water use improve the prediction?*

Both models use 500 decision trees, a minimum leaf size of 5, and are evaluated using out-of-bag predictions — meaning each tree is tested only on data it never saw during training, giving us an honest estimate of generalisation error without needing a separate holdout set. We then compute the out-of-bag $R^2$ for each model, and define:

$$\Delta R^2 = R^2_{anthro} - R^2_{nat}$$

A large positive $\Delta R^2$ means that human abstraction explains a meaningful fraction of water storage variability *beyond* what climate alone can account for. A $\Delta R^2$ near zero means climate is the whole story.

### Identifying the Dominant Driver

To determine *which specific variable* matters most in each basin, we use out-of-bag permutation feature importance — a technique introduced by Breiman in his original 2001 Random Forest paper. The idea is straightforward: for each feature, you randomly shuffle its values across the out-of-bag samples, breaking its relationship with the target. The increase in prediction error tells you how much the model depended on that feature. We extract these importance scores from $M_{anthro}$ for all five predictors — $P$, $ET$, $Q$, $GW_{abs}$, and $SW_{abs}$ — and assign each basin a "dominant driver" based on whichever feature has the highest importance.

### The Technical Challenge: Weak Anthropogenic Signals

There is a serious practical difficulty here that I want to highlight, because it affects any machine learning study that tries to detect human impacts alongside climate signals. Precipitation and evapotranspiration have variance that is *orders of magnitude* larger than groundwater abstraction. $P$ swings by 10–20 centimetres per month across seasons; $GW_{abs}$ changes by fractions of a centimetre. When a decision tree chooses which variable to split on, it maximises variance reduction — so it almost always picks $P$ or $ET$. The abstraction signal gets buried.

Our solution is to set the number of candidate variables at each split to one. In MATLAB's TreeBagger, this is the `NumPredictorsToSample` parameter. With this set to 1, every tree is built by looking at a single randomly chosen predictor at each node. Across hundreds of trees, some will be forced to split on $GW_{abs}$ or $SW_{abs}$ — and if those variables carry any signal at all, it accumulates across the ensemble. This is conceptually analogous to the feature sub-sampling philosophy of Breiman's original algorithm, but pushed to its extreme to amplify weak but physically meaningful signals.

### Validation

Validation is where many hydrological machine learning studies fall short, and it is something reviewers at journals like *Water Resources Research* examine closely. Two specific pitfalls concern us:

**First**, temporal autocorrelation leakage. Standard random K-fold cross-validation shuffles the time indices, allowing January 2010 to train and February 2010 to test — or vice versa. Because hydrological time series are strongly autocorrelated, this inflates skill metrics. We instead use **3-year contiguous block cross-validation**: the 213-month record is split into non-overlapping 36-month blocks, and each block is held out in turn while the model trains on the rest.

**Second**, deseasonalisation leakage. If you compute the monthly climatology from the *entire* time series including the test period, the mean of each calendar month has already "seen" the test data. We prevent this by computing climatologies from the **fixed 2004–2009 baseline** only — never from test blocks.

`[FIGURE: model_comparison_boxplots.png — Side-by-side boxplots of NSE, KGE, and RMSE for M_nat vs M_anthro under block CV]`

### Results

The trend analysis — using the non-parametric Theil-Sen slope estimator for robustness, with significance assessed via the Hamed and Rao (1998) autocorrelation-corrected Modified Mann-Kendall test — reveals widespread TWS decline.

`[FIGURE: tws_basin_trends.png — Global map of TWS trends; blue = gaining, red = declining; stippling on non-significant basins]`

`[FIGURE: top20_negative_trends.png — Top-20 most severely declining basins]`

The attribution results show clear spatial patterns in driver dominance:

`[FIGURE: global_attribution_map.png — Each basin coloured by its dominant driver: P (blue), ET (green), Q (cyan), GW_abs (orange), SW_abs (red)]`

Precipitation dominates in the tropics and high latitudes. Evapotranspiration dominates in semi-arid regions experiencing warming. Groundwater abstraction emerges as the dominant driver in South Asia, the Middle East, and parts of North Africa — basins where the literature independently documents intensive irrigation and aquifer depletion: the Indus (Tiwari et al., 2009; Rodell et al., 2009), the Tigris-Euphrates (Voss et al., 2013; Joodaki et al., 2014), and the North China Plain (Feng et al., 2013).

`[FIGURE: delta_r2_attribution_barchart.png — ΔR² for all 103 basins, sorted, coloured by dominant driver]`

The $\Delta R^2$ chart confirms this quantitatively: the basins with the largest anthropogenic signal — Indus, Tigris-Euphrates — sit at the top. Basins in wet tropical regions, where climate alone explains the variability, cluster near zero.

### Closing

To summarise: we developed a twin machine learning framework that, for the first time, systematically attributes TWS decline across 103 global river basins to either natural hydroclimate variability or human water abstraction. The framework incorporates a forced feature-sampling strategy to amplify weak anthropogenic signals, and a leakage-aware validation protocol that respects the temporal structure of hydrological data. The results provide basin-specific, policy-relevant diagnostics — telling water managers not just *that* a basin is losing water, but *why*, and therefore *what to do about it*.

Thank you. I am happy to take your questions.

---

## References Cited in Speech

| Citation | Full Reference |
|----------|----------------|
| Rodell et al., 2018 | Rodell, M., et al. Emerging trends in global freshwater availability. *Nature*, 557, 651–659. |
| Scanlon et al., 2018 | Scanlon, B. R., et al. Global models underestimate large decadal declining and rising water storage trends. *PNAS*, 115(6), E1080–E1089. |
| Tapley et al., 2004 | Tapley, B. D., et al. GRACE measurements of mass variability in the Earth system. *Science*, 305, 503–505. |
| Breiman, 2001 | Breiman, L. Random forests. *Machine Learning*, 45(1), 5–32. |
| Hersbach et al., 2020 | Hersbach, H., et al. The ERA5 global reanalysis. *QJRMS*, 146(730), 1999–2049. |
| Martens et al., 2017 | Martens, B., et al. GLEAM v3: satellite-based land evaporation and root-zone soil moisture. *GMD*, 10, 1903–1925. |
| Sutanudjaja et al., 2018 | Sutanudjaja, E. H., et al. PCR-GLOBWB 2: a 5 arcmin global hydrological and water resources model. *GMD*, 11, 2429–2453. |
| Hamed & Rao, 1998 | Hamed, K. H. & Rao, A. R. A modified Mann-Kendall trend test for autocorrelated data. *J. Hydrology*, 204, 182–196. |
| Tiwari et al., 2009 | Tiwari, V. M., et al. Dwindling groundwater resources in northern India. *GRL*, 36, L18401. |
| Rodell et al., 2009 | Rodell, M., et al. Satellite-based estimates of groundwater depletion in India. *Nature*, 460, 999–1002. |
| Voss et al., 2013 | Voss, K. A., et al. Groundwater depletion in the Middle East from GRACE. *WRR*, 49(2), 904–914. |
| Joodaki et al., 2014 | Joodaki, G., et al. Estimating the human contribution to groundwater depletion in the Middle East. *WRR*, 50(3), 2679–2692. |
| Feng et al., 2013 | Feng, W., et al. Evaluation of groundwater depletion in North China using GRACE. *WRR*, 49(4), 2110–2118. |
