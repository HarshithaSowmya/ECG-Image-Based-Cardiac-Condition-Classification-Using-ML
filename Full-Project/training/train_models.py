import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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
# Load PCA datasets
# ---------------------------------------

train_df = pd.read_csv("training/train_pca.csv")
test_df = pd.read_csv("training/test_pca.csv")

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


print("Training:", X_train.shape)
print("Testing :", X_test.shape)


# ---------------------------------------
# Models
# ---------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )
}


results = []


# ---------------------------------------
# Train and evaluate
# ---------------------------------------

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

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

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# ---------------------------------------
# Compare models
# ---------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n\nMODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))


# ---------------------------------------
# Save results
# ---------------------------------------

results_df.to_csv(
    "training/model_comparison.csv",
    index=False
)


# ---------------------------------------
# Save the model with highest macro F1
# ---------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

joblib.dump(
    best_model,
    "models/ecg_model.pkl"
)

print("\nSelected model:", best_model_name)
print("Saved to: models/ecg_model.pkl")