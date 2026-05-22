# Mental Health Sentiment Analysis Using Deep Learning

## Project Overview

Mental health issues are increasingly discussed through online platforms such as social media, forums, and support communities. Manually analyzing large volumes of text data is difficult and time-consuming.

This project uses Natural Language Processing (NLP) and Deep Learning techniques to automatically classify mental health-related text into different psychological categories. Multiple neural network architectures were implemented and compared to identify the most effective model.

---

## Description

Mental Health Sentiment Analysis is a Natural Language Processing (NLP) and Deep Learning project that automatically classifies mental health-related text into different psychological categories. The project implements and compares multiple recurrent neural network architectures, including LSTM, Bidirectional LSTM (BiLSTM), GRU, and Stacked BiLSTM, to analyze textual patterns and predict mental health conditions.

The complete pipeline includes data preprocessing, feature engineering, sequence modeling, model evaluation, and prediction. This project demonstrates how deep learning techniques can be applied to mental health text analytics and automated sentiment classification.

---

## Problem Statement

The objective of this project is to develop an automated system that can analyze textual statements and predict the corresponding mental health category.

The system helps in:

- Early identification of mental health concerns
- Automated text classification
- Large-scale sentiment and psychological analysis
- Supporting mental health monitoring systems

---

## Dataset

The project uses a Mental Health Dataset containing textual responses and corresponding mental health labels.

### Dataset Includes

- User text statements
- Mental health categories
- Multiple classes representing emotional and psychological conditions

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-Learn
- TensorFlow
- Keras

### Deep Learning Models

- LSTM
- Bidirectional LSTM (BiLSTM)
- GRU
- Stacked BiLSTM with Dropout

---

## Project Workflow

### 1. Data Collection

Load and inspect the mental health text dataset.

### 2. Data Exploration

- Dataset inspection
- Label distribution analysis
- Missing value checking
- Data visualization

### 3. Text Preprocessing

Text cleaning includes:

- Lowercase conversion
- Removing special characters
- Removing stopwords
- Lemmatization
- Tokenization

### 4. Feature Engineering

- Tokenizer creation
- Vocabulary generation
- Sequence conversion
- Sequence padding

### 5. Data Preparation

- Label Encoding
- Train-Test Split

### 6. Model Building

Implemented multiple Deep Learning architectures:

#### LSTM

Captures long-term dependencies in textual sequences.

#### Bidirectional LSTM (BiLSTM)

Processes information from both forward and backward directions to improve contextual understanding.

#### GRU

Efficient sequence-learning architecture with fewer parameters compared to LSTM.

#### Stacked BiLSTM

Multiple Bidirectional LSTM layers combined with dropout regularization for deeper contextual feature extraction.

### 7. Model Training

Models were trained using:

- Adam Optimizer
- Sparse Categorical Crossentropy Loss
- Accuracy Metric

### 8. Model Evaluation

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## Model Architecture

```text
Input Text
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Sequence Padding
     ↓
Embedding Layer
     ↓
LSTM / BiLSTM / GRU Layers
     ↓
Dense Layers
     ↓
Softmax Output Layer
     ↓
Predicted Mental Health Category
```

---

## Results

Multiple deep learning architectures were trained and evaluated for mental health text classification.

| Model | Accuracy |
|---------|---------|
| LSTM | 69.02% |
| BiLSTM | 68.83% |
| GRU | 63.68% |
| Stacked BiLSTM | 65.00% |

The models successfully learned semantic patterns from mental health-related textual data and demonstrated the effectiveness of recurrent neural networks for text classification tasks.

Although LSTM achieved the highest test accuracy, a Stacked BiLSTM architecture was also developed and deployed to explore deeper bidirectional contextual learning and compare performance across different sequence modeling approaches.

---

## Screenshots

Project screenshots demonstrating:

- Dataset exploration
- Label distribution analysis
- Text preprocessing
- Model architecture
- Training performance
- Classification reports
- Prediction outputs

are available in the **Screenshots** folder.

---

## Applications

- Mental Health Monitoring Systems
- Sentiment Analysis
- Social Media Analysis
- Psychological Support Systems
- Early Risk Detection Systems
- Healthcare Analytics
- Mental Health Research

---

## Installation

Clone the repository:

```bash
git clone <repository-link>
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook Mental_Health_Analysis.ipynb
```

---

## Repository Structure

```text
Mental-Health-Sentiment-Analysis/
│
├── Mental_Health_Analysis.ipynb
├── Mental Health Dataset.csv
├── README.md
├── requirements.txt
├── Saved_model/
│
└── Screenshots/
```

---

## Future Improvements

- Fine-tuning Transformer-based models (BERT, RoBERTa)
- Hyperparameter Optimization
- Explainable AI Techniques
- Real-Time Prediction API
- Deployment using Flask/FastAPI
- Multi-Language Support
- Cloud Deployment

---

## Author

**Gayathri Pasupuleti**