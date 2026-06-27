# ML Classification Projects

Two deep learning classification projects built with TensorFlow/Keras as part of an AI & Machine Learning portfolio.

---

## Projects

### 1.  Rice Leaf Disease Classification (Image Classification)

Classifies rice leaf images into 6 disease categories using Convolutional Neural Networks (CNNs).

**Classes:** Bacterial Leaf Blight, Leaf Blast, Sheath Blight, Brown Spot, Leaf Scald, Healthy Rice Leaf

**Approach:**
- Data preprocessing, augmentation, and corrupted image removal
- Baseline CNN (3 convolutional layers)
- Deeper CNN with batch normalization and dropout regularization
- Transfer learning with pretrained VGG16
  
- Fine-tuned VGG16 with unfrozen layers

**Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

 [`image-classification/`](./image-classification/)

---

### 2.  Fake News Classification (Text Classification)

Binary classification of news articles as real or fake using RNN, LSTM, and pretrained GloVe word embeddings.

**Dataset:** True vs. Fake News Dataset — 20,000 balanced samples (10,000 real, 10,000 fake)

**Approach:**
- Text preprocessing: contraction expansion, lemmatization, stopword removal
- Simple RNN baseline
- LSTM model
- Bidirectional LSTM with pretrained GloVe embeddings

**Evaluation:** Accuracy, Classification Report, Confusion Matrix

 [`text-classification/`](./text-classification/)

---

## Repository Structure

```
ml-classification-projects/
├── README.md
├── image-classification/
│   ├── rice_leaf_disease_classification.ipynb
│   └── report.pdf
└── text-classification/
    ├── fake_news_classification.ipynb
    └── report.pdf
```

---

## Requirements

Install all dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow nltk gensim wordcloud contractions
```

For NLTK resources, run once inside Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## How to Run

Both notebooks were developed and tested in **Google Colab**. They can also be run locally with Jupyter.

### Option A — Google Colab (Recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload the notebook (`File > Upload notebook`)
3. Upload the dataset to your Google Drive and update the dataset path in the notebook
4. Run all cells (`Runtime > Run all`)

**Datasets needed:**
- Image classification: Rice Leaf Disease dataset — place in `./Rice_Leaf_AUG/` (folder with one subfolder per class)
- Text classification: `truevsfakenews.csv` — update the file path in the data loading cell

### Option B — Local (Jupyter Notebook)

1. Clone this repo:
   ```bash
   git clone https://github.com/onikasbarbz/ml-classification-projects.git
   cd ml-classification-projects
   ```
2. Install dependencies (see above)
3. Place the dataset in the correct location (see notebook setup section)
4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

---

**Luniva Shrestha**  

