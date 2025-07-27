import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
a, b, c = 0.5, 0.3, 0.2
threshold = 50.0

# Load data
df = pd.read_csv("device_data.csv")

# Step 1: Calculate fault rate
df["fault_rate"] = a * df["memory"] + b * df["traffic"] + c * df["latency"]
df["faulty"] = df["fault_rate"].apply(lambda x: 1 if x > threshold else 0)

# Step 2: Feature Engineering
df["memory_traffic_ratio"] = df["memory"] / (df["traffic"] + 1)
df["latency_per_traffic"] = df["latency"] / (df["traffic"] + 1)

# Step 3: Define features & target
features = ["memory", "traffic", "latency", "memory_traffic_ratio", "latency_per_traffic"]
X = df[features]
y = df["faulty"]

# Step 4: Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: ML Pipeline with Standard Scaler
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

# Step 6: GridSearch for Logistic Regression
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
grid_search.fit(X_train, y_train)

# Step 7: Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 8: Compare with Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("\n🌲 Random Forest Accuracy:", rf_model.score(X_test, y_test))

# Step 9: Plot Feature Importances (Random Forest)
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=features, y=importances)
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
import joblib

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')

}

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results.append((name, acc, f1, auc))

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "ROC AUC"])
print("\n🔍 Model Comparison:\n", results_df)

# Plot ROC Curves
plt.figure(figsize=(10, 6))
for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the best model (Random Forest in this case)
final_model = models["Random Forest"]
final_model.fit(X, y)
joblib.dump(final_model, "fault_detector_model.pkl")
print("\n💾 Model saved to fault_detector_model.pkl")

# 🔍 Custom Prediction Function
def predict_fault(memory, traffic, latency):
    memory_traffic_ratio = memory / (traffic + 1)
    latency_per_traffic = latency / (traffic + 1)
    input_data = pd.DataFrame([[memory, traffic, latency, memory_traffic_ratio, latency_per_traffic]],
                              columns=features)
    model = joblib.load("fault_detector_model.pkl")
    pred = model.predict(input_data)[0]
    score = model.predict_proba(input_data)[0][1]
    print(f"\n📍 Prediction for input: Memory={memory}, Traffic={traffic}, Latency={latency}")
    print(f"Faulty? {'Yes 🚨' if pred == 1 else 'No ✅'} (Confidence: {score:.2f})")

# Example Prediction
predict_fault(memory=70, traffic=60, latency=30)
