import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall
from imblearn.under_sampling import RandomUnderSampler

# 1. dataset
data = pd.read_csv("DDoSdata.csv")

print("\nLabel cross-check:")
print(pd.crosstab(data['attack'], data['category']))

# Sample some normal/attack rows
print("\nSample attack traffic (attack=1):")
print(data[data['attack'] == 1].head(2)[['saddr', 'daddr', 'bytes', 'pkts']])

print("\nSample normal traffic (attack=0):")
print(data[data['attack'] == 0].head(2)[['saddr', 'daddr', 'bytes', 'pkts']])
# Find potential binary target columns
for col in data.columns:
    unique_vals = data[col].unique()
    if len(unique_vals) == 2:
        print(f"Potential target: {col}")
        print(data[col].value_counts())

        random_samples = data.sample(n=20, random_state=42)
print("\nfraud watch:")
print(random_samples[['saddr', 'daddr', 'bytes', 'pkts', 'attack']])


# 2. Identification of target column 
target_column = 'attack' 
print("All columns:", data.columns.tolist())

# 3. Separate features and labels
X = data.drop(columns=[target_column])
y = data[target_column]
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
print(pd.Series(y_resampled).value_counts())

# 4. Preview
print("\nClass distribution:\n", y.value_counts())

# 5. Encode labels if needed
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Rest of your code remains the same...
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])), 
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]  
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

