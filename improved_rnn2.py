import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('./cleaned_reviews.csv')

sentences = df['review'].values
labels = df['sentiment'].values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

maxlen = 200
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=21)

model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# EarlyStopping adjusted to monitor 'val_accuracy'
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=10, batch_size=128,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f'Accuracy Score: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Epoch 1/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 31s 94ms/step - accuracy: 0.7175 - loss: 0.6302 - val_accuracy: 0.8631 - val_loss: 0.3490
# Epoch 2/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 27s 89ms/step - accuracy: 0.8940 - loss: 0.2945 - val_accuracy: 0.8851 - val_loss: 0.2927
# Epoch 3/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 28s 92ms/step - accuracy: 0.9131 - loss: 0.2454 - val_accuracy: 0.8866 - val_loss: 0.2985
# Epoch 4/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 29s 93ms/step - accuracy: 0.9210 - loss: 0.2242 - val_accuracy: 0.8823 - val_loss: 0.3020
# Epoch 5/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 29s 92ms/step - accuracy: 0.9290 - loss: 0.2051 - val_accuracy: 0.8779 - val_loss: 0.3358
# Epoch 6/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 30s 96ms/step - accuracy: 0.9359 - loss: 0.1885 - val_accuracy: 0.8766 - val_loss: 0.3659
# Epoch 7/10
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 31s 100ms/step - accuracy: 0.9412 - loss: 0.1790 - val_accuracy: 0.8758 - val_loss: 0.3613
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.8896 - loss: 0.2910
# Test Accuracy: 0.8866
# 310/310 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step
# Accuracy Score: 0.8866
# Precision: 0.8743
# Recall: 0.9056
# F1 Score: 0.8897
