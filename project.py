import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Coefficients (used for labeling)
a, b, c = 0.5, 0.3, 0.2
threshold = 50

# Load and prepare data
df = pd.read_csv("device_data.csv")

# Calculate fault rate
df["fault_rate"] = a * df["memory"] + b * df["traffic"] + c * df["latency"]

# Create binary target for ML
df["faulty"] = df["fault_rate"].apply(lambda x: 1 if x > threshold else 0)

# Exploratory Data Analysis
def plot_distribution():
    sns.pairplot(df[['memory', 'traffic', 'latency', 'faulty']], hue="faulty")
    plt.suptitle("Feature Distribution", y=1.02)
    plt.show()

# Features & target
X = df[['memory', 'traffic', 'latency']]
y = df["faulty"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n🔍 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Visualize fault rates
def plot_fault_rate_bar():
    plt.figure(figsize=(10, 6))
    sns.barplot(x='device', y='fault_rate', data=df, hue='faulty', palette='Set2')
    plt.axhline(y=threshold, color='red', linestyle='--', label="Fault Threshold")
    plt.title("Device Fault Rates")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_fault_rate_bar()
