# Movie Review Sentiment Classification  

Natural Language Processing with TF-IDF, LightGBM, and BERT Embeddings  
**Author: Rhiannon Fillingham**

---

## Overview  

The Film Junky Union wants to automate the process of filtering large volumes of movie reviews.  
Using an IMDB-based dataset with labeled sentiment, I built several machine learning models to classify reviews as **positive or negative**, with a performance target of **F1 ≥ 0.85** on the test set.

This project demonstrates text preprocessing, exploratory analysis, TF-IDF vectorization, classical machine learning, gradient boosting, and transformer-based embeddings (BERT).

---

## Dataset  

**Source:** IMDB reviews (Maas et al., ACL 2011)  
**File:** `imdb_reviews.tsv`

**Key fields:**  

- `review` — review text  
- `pos` — sentiment label (`1` = positive, `0` = negative)  
- `ds_part` — train or test designation  

Additional fields such as `start_year`, `tconst`, and `votes` were explored during EDA.

---

## Approach  

### 1. Exploratory Data Analysis  

- Examined sentiment balance across train/test sets  
- Explored review lengths and text distribution  
- Analyzed review volume by year and movie  

### 2. Text Preprocessing  

- Lowercasing  
- Removing digits and punctuation  
- Tokenization (NLTK)  
- Lemmatization (spaCy)  
- Train/test separation using `ds_part`  

### 3. Feature Engineering  

- TF-IDF vectorization (unigrams + bigrams)  
- Lemmatized text vectorization  
- BERT embeddings via a pre-trained transformer model  

### 4. Models Trained  

- Logistic Regression (TF-IDF baseline)  
- Logistic Regression (lemmatized TF-IDF)  
- LightGBM Classifier (TF-IDF)  
- Logistic Regression on BERT embeddings  

### 5. Evaluation Metrics  

- F1-score (primary metric)  
- ROC-AUC  
- Average Precision Score  

---

## Results  

| Model | F1 Score | ROC-AUC | APS | Notes |
|-------|----------|---------|---------|-------|
| TF-IDF + Logistic Regression | ~0.88 | ~0.95 | ~0.95 | Strong, simple baseline |
| Lemmatized TF-IDF + Logistic Regression | ~0.88 | ~0.95 | ~0.95 | Similar performance |
| TF-IDF + LightGBM | ~0.88 | ~0.95 | ~0.95 | Better probability ranking |
| **BERT Embeddings + Logistic Regression** | **~0.89+** | **Highest** | **Highest** | Most nuanced and consistent |

**Final Model:**  
**BERT embeddings + Logistic Regression** delivered the strongest overall performance while maintaining interpretability and speed on reduced samples.

All models achieved or exceeded the required **F1 ≥ 0.85**.

---

## Example Predictions  

**Review:**  
“Surprisingly thoughtful and emotional. I didn’t expect to enjoy it this much.”  
**Prediction:** Positive  

**Review:**  
“This movie dragged endlessly, and the plot made absolutely no sense.”  
**Prediction:** Negative  

Testing custom reviews confirms consistent model behavior across writing styles.

---

## Tech Stack  

- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- spaCy  
- LightGBM  
- Transformers (HuggingFace)  
- PyTorch  
- Matplotlib, Seaborn  

---

## Environment Setup  

You can recreate the project environment using either **Conda** or **pip**.

### Option 1: Conda (recommended)  

From the project root:

```bash
conda env create -f environment.yml
conda activate nlp-sentiment-env
python -m spacy download en_core_web_sm
```

For NLTK stopwords (used in preprocessing):

```python
import nltk
nltk.download("stopwords")
```

### Option 2: pip

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

And in Python:

```python
import nltk
nltk.download("stopwords")
```

---

## Run the Notebook

Once your environment is activated and dependencies are installed, launch Jupyter Notebook from the project root:

```bash
jupyter notebook notebooks/nlp_sentiment_analysis.ipynb
```

---

## Repository Structure

```
project/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   └── imdb_reviews.tsv
│
├── notebooks/
│   └── nlp_sentiment_analysis.ipynb
│
└── .gitignore
```

---

## Future Improvements

- Fine-tune a transformer model directly for full-text classification

- Explore hyperparameter optimization for LightGBM and logistic regression

- Improve preprocessing with profanity and emoji normalization

- Package the model into an API endpoint for deployment

---

**Author: Rhiannon Fillingham**
