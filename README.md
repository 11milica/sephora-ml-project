#  Data Preprocessing Summary — Notebook 00

In this notebook, I handled all data ingestion, merging, cleaning, and feature engineering necessary to prepare the dataset for downstream machine learning tasks. I deliberately structured this as a **dedicated preprocessing pipeline** so that I wouldn't need to repeat cleaning logic in every modeling notebook. Everything in this notebook is reusable, traceable, and built for scale.

---

##  Step 1 – Data Ingestion

The Kaggle dataset included:
- Over 1,094,000 user reviews across five CSV chunks
- 8,494 product records with brand, price, category, ingredients, and more

I wrote a custom function to concatenate all review chunks and merged them with the product metadata using `product_id`.

> I used `dtype` typing (e.g. treating brand_name as categorical) to reduce memory usage and enforce column consistency.

---

##  Step 2 – Data Cleaning

I created a flexible cleaning function that:
- Drops rows with missing `review_text` or `rating`
- Fills missing numerical fields (`price_usd`, `loves_count`) using the **median**
- Fills missing categorical fields (`brand_name`, `primary_category`) using `'Unknown'`
    - If a column is of type `category`, I first add `'Unknown'` as a valid category
    - I replaced `pd.api.types.is_categorical_dtype()` with the modern `isinstance(..., pd.CategoricalDtype)` to avoid deprecation

All changes are made using `.copy()` and `.loc[]` to avoid chained assignment warnings and to ensure clean memory-safe operations.

---

##  Step 3 – Sentiment Labeling

I excluded reviews with a neutral rating of 3 and defined sentiment as:
- `1` for ratings 4 and 5
- `0` for ratings 1 and 2

> This binary labeling prepares the dataset for classification using artificial neural networks in the next stage.

---

##  Step 4 – Feature Engineering

I engineered several new features:
- `review_length` (number of characters)
- `word_count` (number of tokens)
- `price_bin` (price grouped into 5 quantiles using `pd.qcut`)

> These features will help improve model interpretability and performance across various ML tasks (like trend analysis and recommendation).

---

##  Step 5 – Text Preprocessing

I implemented a robust `preprocess_text()` function that:
- Lowercases text
- Removes HTML tags and non-alphabetic characters
- Filters out short words and stopwords (using NLTK)
- Is wrapped in a try/except block to gracefully handle unexpected text
- Uses `tqdm.progress_apply()` to monitor progress on large datasets

> I had to explicitly install and enable `tqdm` and NLTK. I also registered the progress bar using `tqdm.pandas()`.

---

##  Environment & Kernel Troubleshooting

At first, I encountered `ModuleNotFoundError: No module named 'nltk'`, even though I had installed it. I realized that **Jupyter was running a different Python environment** than the one where I installed my packages.

To fix this, I created a custom kernel for the correct environment:

```bash
python -m ipykernel install --user --name sephora_env --display-name "Python (sephora_env)"

Then I selected the new kernel from Kernel → Change Kernel in JupyterLab. This resolved the issue completely.

## Step 6 – Exporting Cleaned Data
I exported two outputs to the ../data/processed/ directory:

full_dataset.parquet — 1,011,215 fully cleaned reviews with sentiment labels and engineered features
✅ Saved using pyarrow for efficient storage

product_catalog.csv — 2,347 deduplicated products with brand, price, and category
✅ For use in the KNN recommendation system later