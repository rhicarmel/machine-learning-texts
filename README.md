# 🗂️ Machine Learning for Texts | Data Science Project
### Sentiment Classification of Movie Reviews  
<div align="center">

[![Open Notebook](https://img.shields.io/badge/View_Notebook-8A2BE2?style=for-the-badge)](./notebooks/nlp_sentiment_analysis.ipynb)
[![Dataset](https://img.shields.io/badge/Download_Dataset-IMDb_Reviews-4B8B3B?style=for-the-badge)](https://ai.stanford.edu/~amaas/data/sentiment/)

</div>

## Project Overview

This project builds, compares, and evaluates multiple machine learning models for **binary sentiment classification** on IMDb-style movie reviews.  
The goal is to determine whether a review expresses a **positive** or **negative** opinion using both classical machine learning and modern NLP methods.

**Included in the workflow:**

- Text normalization and preprocessing  
- spaCy lemmatization  
- TF-IDF feature engineering  
- Classical models (Logistic Regression, LightGBM)  
- Deep contextual embeddings (BERT)  
- A unified evaluation pipeline (ROC, PRC, F1 curves)  
- Custom review predictions for real-world testing  

---

## 📁 Dataset

This project uses the IMDb Reviews dataset (Maas et al., 2011).

🔗 **Download dataset (direct link):**  
https://ai.stanford.edu/~amaas/data/sentiment/

Place the file `imdb_reviews.tsv` into the `./data/` folder before running the notebook.

**Key fields:**  

- `review` — review text  
- `pos` — sentiment label (`1` = positive, `0` = negative)  
- `ds_part` — train or test split  
- Additional metadata (`start_year`, `tconst`, `votes`) was explored during EDA.

---

## 📊 Results  

| Model | F1 Score | ROC-AUC | APS | Notes |
|-------|----------|---------|---------|-------|
| TF-IDF + Logistic Regression | ~0.88 | ~0.95 | ~0.95 | Strong baseline |
| Lemmatized TF-IDF + Logistic Regression | ~0.88 | ~0.95 | ~0.95 | Similar performance |
| TF-IDF + LightGBM | ~0.88 | ~0.95 | ~0.95 | Better probability calibration |
| BERT Embeddings + Logistic Regression (sampled) | ~0.81 | High | High | Strong contextual modeling |

**Final Model (Full Dataset):**  
**🟦 Model 4 — LightGBM + TF-IDF + spaCy Lemmatization**  
Delivered the strongest overall performance on the full dataset.

All models achieved or exceeded the required **F1 ≥ 0.85**.

---

## 🔍 Example Predictions  

Below are sample predictions from **🟦 Model 4**, the top full-dataset model:

| Review | Probability Positive | Sentiment |
|--------|----------------------|-----------|
| “I was really fascinated with the movie.” | 0.60 | Positive |
| “I fell asleep in the middle of the movie.” | 0.47 | Negative |
| “What a rotten attempt at a comedy.” | 0.15 | Negative |
| “Launching on Netflix was a brave move…” | 0.73 | Positive |

The model performs well on both clear and nuanced sentiment.

---

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-026AA7?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-00A64F?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C78A8?style=for-the-badge)

</div>

---

## ⚙️ Environment Setup  

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

## ▶️ Run the Notebook

Once your environment is activated and dependencies are installed, launch Jupyter Notebook from the project root:

```bash
jupyter notebook notebooks/nlp_sentiment_analysis.ipynb
```

---

## 📂 Repository Structure

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

## 🌱 Future Improvements

- Fine-tune a transformer model directly for full-text classification

- Explore hyperparameter optimization for LightGBM and logistic regression

---

**👩🏻‍💻 Author: Rhiannon Fillingham**
<p align="center">

</p> <p align="center"> <a href="https://www.linkedin.com/in/rhiannon-fillingham"><img src="https://img.shields.io/badge/LinkedIn-Rhiannon_Fillingham-blue?logo=linkedin" /></a> <a href="mailto:rhiannon.filli@gmail.com"><img src="https://img.shields.io/badge/Email-rhiannon.filli%40gmail.com-red?logo=gmail" /></a> </p>
