import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# ---------------------------------------
# 1. Load dataset
# ---------------------------------------

df = pd.read_csv("training/ecg_features.csv")

feature_columns = [
    column for column in df.columns
    if column.startswith("feature_")
]

X = df[feature_columns]
y = df["label"]


print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ---------------------------------------
# 2. Stratified train/test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ---------------------------------------
# 3. PCA
# ---------------------------------------

pca = PCA(
    n_components=0.95,
    svd_solver="full"
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


print("\nPCA components:", pca.n_components_)
print("Train after PCA:", X_train_pca.shape)
print("Test after PCA:", X_test_pca.shape)

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)


# ---------------------------------------
# 4. Save PCA
# ---------------------------------------

joblib.dump(
    pca,
    "models/pca.pkl"
)


# ---------------------------------------
# 5. Save processed train/test data
# ---------------------------------------

train_data = pd.DataFrame(X_train_pca)
train_data["label"] = y_train.to_numpy()

test_data = pd.DataFrame(X_test_pca)
test_data["label"] = y_test.to_numpy()

train_data.to_csv(
    "training/train_pca.csv",
    index=False
)

test_data.to_csv(
    "training/test_pca.csv",
    index=False
)


print("\nSaved:")
print("models/pca.pkl")
print("training/train_pca.csv")
print("training/test_pca.csv")