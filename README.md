# Feature Engineering – NBA Player Longevity Prediction

## Project Overview
This project focuses on transforming raw NBA player performance statistics into a clean, machine-learning-ready dataset. The primary goal is to engineer predictive features that help forecast whether an NBA player will have a career lasting 5 years or longer. 

## Dataset Description
The dataset (`0f484464-3cc8-4bfc-968b-b1a9fc4d4b1d.csv`) contains historical performance statistics for NBA rookies. 
* [cite_start]**Target Variable:** `target_5yrs` is a binary classification target indicating if the player's career lasted 5 or more years (1 = Yes, 0 = No)[cite: 1].
* [cite_start]**Predictors:** Various traditional per-game statistics, including minutes played (`min`), points (`pts`), rebounds (`oreb`, `dreb`, `reb`), assists (`ast`), steals (`stl`), blocks (`blk`), and turnovers (`tov`)[cite: 1].

## Feature Engineering Workflow
To ensure our machine learning models perform optimally without bias or multicollinearity, the following data processing pipeline was implemented:

1.  [cite_start]**Target Isolation:** The `target_5yrs` column was extracted and isolated early to prevent accidental data leakage during transformations[cite: 1].
2.  [cite_start]**Noise Removal:** Non-predictive identifiers, specifically the index column (`Unnamed: 0`) and player name (`name`), were dropped to prevent the model from memorizing specific players instead of learning generalizable statistical patterns[cite: 1].
3.  **Redundancy & Multicollinearity Handling:** Highly correlated features (Pearson correlation >= 0.90) were removed. We dropped `fgm`, `fga`, `ftm`, `3pa`, and `reb` to simplify the model and reduce variance.
4.  **Composite Metric Creation:** We engineered advanced analytics metrics to better capture player impact:
    * **Points Per Minute (`pts_per_min`)**: Normalizes scoring output based on playing time.
    * **Efficiency Rating (`efficiency_rating`)**: A PER-style approximation combining total positive box score contributions minus turnovers, normalized per minute.
    * **Assist-to-Turnover Ratio (`ast_tov_ratio`)**: Evaluates ball security and playmaking efficiency.
    * **True Shooting Approximation (`true_shooting_approx`)**: Approximates scoring efficiency by factoring in the varying weight of two-pointers, three-pointers, and free throws.
5.  **Data Cleaning & Imputation:** Infinite values generated during division operations were converted to nulls. Missing numeric values were then filled using median imputation, which is robust to outliers and prevents skewed averages from biasing the dataset.

## How to Run
Ensure you have `pandas` and `numpy` installed. Open `nba_feature_engineering.ipynb` in a Jupyter environment and run all cells. The final output is a clean pandas DataFrame ready for supervised learning models like Logistic Regression or Random Forests.