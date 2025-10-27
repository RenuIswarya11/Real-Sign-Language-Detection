import os
import numpy as np
import json
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical  # updated import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# --- Suppress TF warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# --- Paths ---
DATA_PATH = "../data"
MODEL_PATH = "../models"
os.makedirs(MODEL_PATH, exist_ok=True)

# --- Load data ---
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

# --- Preprocess ---
scaler = StandardScaler()
X = scaler.fit_transform(X)  # normalize landmarks

# --- Split dataset for small dataset (no separate validation) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert labels to categorical
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# --- Build simple MLP model ---
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- Callbacks ---
ckpt = callbacks.ModelCheckpoint(
    os.path.join(MODEL_PATH, "best_model.h5"),
    save_best_only=True,
    monitor="accuracy"  # monitor train accuracy for small dataset
)
es = callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor="accuracy"
)

# --- Train ---
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=16,
    callbacks=[ckpt, es]
)

# --- Evaluate ---
y_pred = np.argmax(model.predict(X_test), axis=1)
with open(os.path.join(DATA_PATH,"label_map.json")) as f:
    label_names = json.load(f)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save final model and scaler ---
model.save(os.path.join(MODEL_PATH, "final_model.h5"))
np.save(os.path.join(MODEL_PATH, "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(MODEL_PATH, "scaler_scale.npy"), scaler.scale_)

print("\n✅ Training completed. Model saved in 'models/' folder.")
