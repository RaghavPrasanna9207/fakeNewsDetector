import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# LIME import
from lime.lime_text import LimeTextExplainer

# -------------------------------------------
# Data Load + Sampling for Speed
# -------------------------------------------

def load_data():
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')

    true_df['label'] = 1
    fake_df['label'] = 0

    df = pd.concat([true_df, fake_df])
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    df = df.sample(frac=0.4, random_state=42)      # use a fraction of the data for speed

    return df

# -------------------------------------------
# Text Cleaning
# -------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -------------------------------------------
# Main
# -------------------------------------------

data = load_data()
data['text'] = data['title'] + " " + data['text']
data['text'] = data['text'].apply(clean_text)

X_raw = data['text'].tolist()  # store raw version for LIME
X = data['text'].values
y = data['label'].values

# -------------------------------------------
# Tokenizer + Padding
# -------------------------------------------

MAX_VOCAB = 5000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=MAX_LEN)

# -------------------------------------------
# Train/Test Split
# -------------------------------------------

X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
    X, y, X_raw, test_size=0.2, random_state=42
)

# -------------------------------------------
# Model
# -------------------------------------------

model = Sequential()
model.add(Embedding(input_dim=MAX_VOCAB, output_dim=64, input_length=MAX_LEN))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# -------------------------------------------
# Train
# -------------------------------------------

model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.2)

# -------------------------------------------
# Evaluate
# -------------------------------------------

y_pred = (model.predict(X_test) > 0.5).astype("int32")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------------------
# LIME Explainer
# -------------------------------------------

class_names = ['Fake', 'Real']

def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return model.predict(padded)

explainer = LimeTextExplainer(class_names=class_names)

idx = 0  
example_text = X_raw_test[idx]

print(f"\nExplaining text sample {idx}:\n")
print(example_text)
explanation = explainer.explain_instance(example_text, predict_proba, num_features=10)
explanation.save_to_file('lime_explanation.html')
print("\nExplanation saved to lime_explanation.html")
