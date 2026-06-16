# Feature Engineering – NBA Player Longevity Prediction

## Project Dataset & Objective
The dataset (`0f484464-3cc8-4bfc-968b-b1a9fc4d4b1d.csv`) contains statistical baseline performance indicators of rookie NBA basketball players. The objective is to apply standard data preprocessing, feature selection, metric extraction, and scaling methods to construct a predictive feature map for machine learning classifiers. The target metric is a binary indicator column named `target_5yrs` (reflecting if a player's professional longevity reaches 5+ years).

## Feature Engineering Pipeline Workflow

### 1. Data Exploration & Target Class Analysis
* To prevent leakage, the dependent variable `target_5yrs` was mapped and extracted into its own standalone vector prior to feature transformations.
* Introduced a visual distribution check using a seaborn count bar chart to test for target class imbalances. This ensures model performance metrics (like accuracy vs F1-score) can be evaluated reliably during modeling.

### 2. Identifier Noise Removal
* High-cardinality nominal parameters (`name`) and arbitrary sequence identifiers (`Unnamed: 0`) were dropped from the dataset to prevent the model from over-indexing on non-predictive statistical noise.

### 3. Correlation & Multicollinearity Mitigation
* Calculated a complete Pearson Correlation Coefficient matrix and plotted a visual heatmap to expose structural redundancies.
* To avoid variance instability in linear models, multi-collinear attributes exhibiting cross-correlations where $|r| \geq 0.90$ were systematically pruned. Pruned features include: `fgm`, `fga`, `ftm`, `3pa`, and `reb`.

### 4. Advanced Composite Feature Extraction
Four compound basketball metrics were engineered to measure per-minute volume, efficiency, and execution safety:
* **Points Per Minute (`pts_per_min`)**: Evaluates basic raw scoring density against playing time.
* **Efficiency Rating (`efficiency_rating`)**: A modified PER box-score metric combining total baseline contributions (`pts` + `oreb` + `dreb` + `ast` + `stl` + `blk` - `tov`) normalized per minute.
* **Assist-to-Turnover Ratio (`ast_tov_ratio`)**: Quantifies playmaking decision quality and control.
* **True Shooting Approximation (`true_shooting_approx`)**: Approximates global scoring accuracy factoring in weighted scoring weights of three-point fields, standard fields, and free throws.

### 5. Data Cleaning and Imputation
* Checked for zero-division edge cases by replacing all operational infinite value flags (`inf`, `-inf`) with null references (`NaN`).
* Cleaned missing numeric feature elements using median imputation, protecting the final dataset against the skewing effects of outliers.

### 6. Feature Transformation (Standard Scaling)
* Implemented feature standardization via `StandardScaler`. This centers all predictive feature inputs to a zero mean ($\mu = 0$) and scales them to a uniform unit variance ($\sigma = 1$). 
* This step is crucial for linear estimators (such as Logistic Regression) and distance metrics (such as Support Vector Classifiers) to prevent high-magnitude metrics from structurally dominating optimization steps.
