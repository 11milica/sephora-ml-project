## Project Overview

This repository explores the use of machine learning to predict product ratings from Sephora skincare reviews.
The goal is to build a regression-based model that interprets textual customer feedback and outputs a numerical rating.
The pipeline begins with thorough data preprocessing and culminates in performance evaluation using visual diagnostics and error metrics.
The project is structured into modular notebooks to clearly separate data preparation, model development, and future refinement stages.

## Project Notebooks

This repository is organized into modular notebooks that reflect distinct phases of the machine learning pipeline:

### 00_data_preparation.ipynb

Focuses on loading, inspecting, and cleaning the Sephora skincare review dataset.

- Combined multiple Excel files into a unified dataframe
- Applied text preprocessing techniques (tokenization, lemmatization)
- Mapped sentiment labels and handled missing data
- Prepared structured features for modeling

This notebook establishes a reproducible and consistent base for downstream modeling tasks.

---

### 01_model_evaluation.ipynb

Trains and evaluates an artificial neural network (ANN) regression model to predict product ratings.

- Built and trained a neural network using processed features
- Evaluated performance via mean absolute error (MAE) and bias
- Explored error distribution through residual scatterplots and histograms
- Conducted class-wise error analysis to uncover systematic overestimation patterns

This notebook provides a detailed assessment of model behavior and identifies improvement opportunities for future iterations.

---

Additional notebooks (e.g., 02_model_improvement.ipynb) will expand on architecture tuning, feature exploration, interpretability techniques, and alternative model benchmarks.
