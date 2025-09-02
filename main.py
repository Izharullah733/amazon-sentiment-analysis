import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import optuna
import wandb
# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') 
stop_words = set(stopwords.words('english'))

# Dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
# Download Reviews.csv and place it in the project directory
df = pd.read_csv('Reviews.csv')

# --- Data Preprocessing ---
# Select relevant columns and drop missing values
df = df[['Text', 'Score']].dropna()

# Map scores to sentiments (1-2: Negative (0), 3: Neutral (1), 4-5: Positive (2))
def map_sentiment(score):
    if score <= 2:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df['Sentiment'] = df['Score'].apply(map_sentiment)
df = df.drop('Score', axis=1)

# Clean text: remove special characters, convert to lowercase, remove stopwords
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['Text'] = df['Text'].apply(clean_text)

# Sample 10,000 reviews for computational efficiency
df = df.sample(n=10000, random_state=42)

# Check class distribution
print("Class Distribution:\n", df['Sentiment'].value_counts())

# Split data
X = df['Text']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Engineering ---
# TF-IDF for Logistic Regression baseline
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf_smote, y_train_tfidf_smote = smote.fit_resample(X_train_tfidf, y_train)
print("After SMOTE, class distribution:", np.bincount(y_train_tfidf_smote))

# BERT Dataset
class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ReviewsDataset(X_train, y_train, tokenizer)
test_dataset = ReviewsDataset(X_test, y_test, tokenizer)

# --- Feature Importance Analysis ---
lr = LogisticRegression(multi_class='multinomial', random_state=42)
lr.fit(X_train_tfidf_smote, y_train_tfidf_smote)
feature_importance = pd.Series(lr.coef_[1], index=tfidf.get_feature_names_out()).sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('Top 10 Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
plt.show()

# --- Model Training and Evaluation ---
# Baseline: Logistic Regression
results = []
lr.fit(X_train_tfidf_smote, y_train_tfidf_smote)
y_pred_lr = lr.predict(X_test_tfidf)
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr, average='weighted'),
    'Recall': recall_score(y_test, y_pred_lr, average='weighted'),
    'F1 Score': f1_score(y_test, y_pred_lr, average='weighted'),
    'ROC AUC': roc_auc_score(y_test, lr.predict_proba(X_test_tfidf), multi_class='ovr')
})
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_lr))

# Fine-Tuned BERT with Optuna
def objective(trial):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=trial.suggest_int('epochs', 2, 5),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=trial.suggest_float('lr', 1e-5, 5e-5, log=True),
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    return f1_score(y_test, y_pred, average='weighted')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_params
print("Best BERT Parameters:", best_params)

# Train final BERT model with best parameters
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=best_params['epochs'],
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=best_params['lr'],
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()
predictions = trainer.predict(test_dataset)
y_pred_bert = np.argmax(predictions.predictions, axis=1)
results.append({
    'Model': 'Fine-Tuned BERT',
    'Accuracy': accuracy_score(y_test, y_pred_bert),
    'Precision': precision_score(y_test, y_pred_bert, average='weighted'),
    'Recall': recall_score(y_test, y_pred_bert, average='weighted'),
    'F1 Score': f1_score(y_test, y_pred_bert, average='weighted'),
    'ROC AUC': roc_auc_score(y_test, predictions.predictions, multi_class='ovr')
})
cm_bert = confusion_matrix(y_test, y_pred_bert)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Fine-Tuned BERT')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("\nClassification Report for Fine-Tuned BERT:\n", classification_report(y_test, y_pred_bert))

# --- Model Comparison ---
results_df = pd.DataFrame(results)
print("\nModel Comparison:\n", results_df)

# Visualize model performance
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
for i, metric in enumerate(metrics):
    sns.barplot(x=metric, y='Model', data=results_df, ax=axes[i])
    axes[i].set_title(metric)
plt.tight_layout()
plt.show()
