# **Sentiment Analysis of Amazon Product Review**
**Overview**

This project implements a sentiment analysis model to classify Amazon Fine Food Reviews as positive, negative, or neutral using machine learning and deep learning techniques. It compares a baseline Logistic Regression model with TF-IDF features against a fine-tuned BERT model optimized with Optuna. The project demonstrates advanced NLP skills, including text preprocessing, feature engineering, and model tuning, making it relevant for e-commerce and customer feedback analysis in industrial applications.

**Dataset**

Source: Kaggle - Amazon Fine Food Reviews

Description: Contains ~568,000 reviews of food products from Amazon, with columns Text (review text) and Score (1-5 star rating). Scores are mapped to sentiments (1-2: Negative, 3: Neutral, 4-5: Positive).

**Note:** Download Reviews.csv and place it in the project directory or update the file path in the script.

**Project Structure**

**main.py:** Main script for data preprocessing, feature engineering, model training (Logistic Regression and Fine-Tuned BERT), evaluation, and visualization.



**requirements.txt:** Lists required Python libraries.



**.gitignore:** Specifies files and directories to exclude from version control.



**README.md:** This file, providing project details and instructions.

**Prerequisites**

Python: Version 3.11.9 (or compatible version)

Libraries:

pandas, numpy, seaborn, matplotlib, Scikit-learn, imblearn, transformers, torch, optuna, nltk and wandb.

**Note:** Use a GPU-enabled environment (e.g., Google Colab) for faster BERT training. If prompted for a wandb API key, disable wandb by adding import os; os.environ["WANDB_MODE"] = "disabled" at the top of the script, or sign up at wandb.ai for logging.

**Usage**

**Data Preprocessing:**

Cleans text by removing special characters, converting to lowercase, and removing stopwords using NLTK.

Maps star ratings (1-5) to sentiments (Negative: 0, Neutral: 1, Positive: 2).

Samples 10,000 reviews for computational efficiency and drops missing values.

**Feature Engineering:**

Generates TF-IDF vectors (max 5000 features) for Logistic Regression.

Applies SMOTE to balance classes for the baseline model.

Tokenizes text with BertTokenizer for BERT input.

Analyzes feature importance using Logistic Regression coefficients.

**Models:**

**Logistic Regression:** Baseline model trained on TF-IDF features with SMOTE.

**Fine-Tuned BERT:** Optimized with Optuna for learning rate and epochs, using PyTorch and the Hugging Face Trainer API.

**Visualizations**:

Feature importance bar plot.

Confusion matrices for both models.

Bar plots comparing model performance (Accuracy, Precision, Recall, F1 Score, ROC AUC).

**Results**

**Model Performance**

The performance of the models based on a sample of 10,000 reviews:

**Logistic Regression:** Achieves 78% accuracy with moderate precision (0.62), recall (0.66), and F1-score (0.64), providing a solid baseline.



**Fine-Tuned BERT:** Outperforms with 88% accuracy, higher precision (0.74), recall (0.78), and F1-score (0.76), demonstrating the power of contextual embeddings in sentiment analysis.

**Key Factors Driving Sentiment**

Based on Logistic Regression coefficients, top words influencing sentiment include:

Positive: "great", "good", "excellent"

Negative: "terrible", "bad", "poor"

**Future Improvements**

Increase the dataset size or use stratified sampling for better representation of minority classes (e.g., Neutral).

Experiment with additional BERT variants (e.g., bert-large-uncased) or other transformer models (e.g., RoBERTa).

Implement cross-validation to ensure robust model evaluation.

Add advanced feature engineering, such as n-grams or word embeddings beyond TF-IDF



