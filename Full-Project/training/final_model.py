import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------
# Load train/test PCA data
# ---------------------------------------

train_df = pd.read_csv("training/train_pca.csv")
test_df = pd.read_csv("training/test_pca.csv")

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


# ---------------------------------------
# Final selected model
# ---------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)


# ---------------------------------------
# Train on ALL training data
# ---------------------------------------

print("Training final SVM...")

model.fit(X_train, y_train)


# ---------------------------------------
# Final test evaluation
# ---------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {f1:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "NORMAL",
            "AHB",
            "MI",
            "PMI"
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ---------------------------------------
# Save final model
# ---------------------------------------

joblib.dump(
    model,
    "models/ecg_model.pkl"
)

print("\nFinal model saved:")
print("models/ecg_model.pkl")