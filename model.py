import pandas as pd
import numpy as np
import pickle, json, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("upi_fraud_engineered.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Fraud rate:", round(df['FraudFlag'].mean() * 100, 2), "%")

# ── Booleans to int ───────────────────────────────────────────
for col in ["IsNewDevice", "IsUnusualLocation", "IsNightHour", "FraudFlag"]:
    df[col] = df[col].astype(int)

# ── Encode categoricals ───────────────────────────────────────
encoders = {}
for col in ["SenderBank", "MerchantCategory", "TransactionType", "DeviceType", "SenderState"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Features and Target ───────────────────────────────────────
X = df.drop(columns=["FraudFlag"])
y = df["FraudFlag"]

print("\nFeatures:", X.columns.tolist())
print("Class distribution:", y.value_counts().to_dict())

# ── Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── SMOTE ─────────────────────────────────────────────────────
smote = SMOTE(random_state=42, sampling_strategy=1.0)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE → Fraud: {y_train_res.sum()} | Legit: {(y_train_res==0).sum()}")

# ── Build ANN ─────────────────────────────────────────────────
model = Sequential([
    Input(shape=(X_train_res.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Train ─────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss', min_lr=1e-6)
]

print("\n🚀 Training started...")
history = model.fit(
    X_train_res, y_train_res,
    epochs=50,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ── Threshold Tuning ──────────────────────────────────────────
y_pred_prob = model.predict(X_test).flatten()
thresholds  = np.arange(0.1, 0.9, 0.05)
best_thresh, best_f1 = 0.5, 0.0

print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>6}")
print("-" * 55)
for t in thresholds:
    yp  = (y_pred_prob >= t).astype(int)
    p   = precision_score(y_test, yp, zero_division=0)
    r   = recall_score(y_test, yp, zero_division=0)
    f   = f1_score(y_test, yp, zero_division=0)
    cmt = confusion_matrix(y_test, yp)
    print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f} {cmt[1][1]:>5} {cmt[0][1]:>6}")
    if f > best_f1:
        best_f1, best_thresh = f, t

print(f"\n✅ Best threshold: {best_thresh:.2f} | Best F1: {best_f1:.4f}")

# ── Final Evaluation ──────────────────────────────────────────
y_pred = (y_pred_prob >= best_thresh).astype(int)
acc    = accuracy_score(y_test, y_pred)
prec   = precision_score(y_test, y_pred, zero_division=0)
rec    = recall_score(y_test, y_pred, zero_division=0)
f1     = f1_score(y_test, y_pred, zero_division=0)
auc    = roc_auc_score(y_test, y_pred_prob)
cm     = confusion_matrix(y_test, y_pred)

print("\n" + "="*45)
print("      FINAL EVALUATION METRICS")
print("="*45)
print(f"  Threshold : {best_thresh:.2f}")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
print("\n" + classification_report(y_test, y_pred))

# ── Save Artifacts ────────────────────────────────────────────
model.save("fraud_ann_model.keras")
with open("scaler.pkl",        "wb") as f: pickle.dump(scaler, f)
with open("encoders.pkl",      "wb") as f: pickle.dump(encoders, f)
with open("feature_names.pkl", "wb") as f: pickle.dump(list(X.columns), f)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
metrics = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "roc_auc": round(auc, 4),
    "best_threshold": round(float(best_thresh), 2),
    "confusion_matrix": cm.tolist(),
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "train_acc": history.history["accuracy"],
    "val_acc": history.history["val_accuracy"],
    "total_transactions": len(df),
    "fraud_count": int(y.sum()),
    "legit_count": int((y == 0).sum()),
    "fraud_rate": round(float(y.mean()) * 100, 2),
    "y_test": y_test.tolist(),
    "y_probs": y_pred_prob.tolist()
}
with open("model_metrics.json", "w") as f: json.dump(metrics, f)

print("\n✅ All artifacts saved!")
print("   fraud_ann_model.keras")
print("   scaler.pkl | encoders.pkl | feature_names.pkl")
print("   model_metrics.json")
print("\nNext: streamlit run app.py")