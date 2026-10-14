import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_FILE = "data/processed/training_data.csv"


FEATURES = [
    "join_count",
    "aggregation",
    "filter",
    "group_by",
    "tables_used"
]

TARGET = "best_schema"


def train_model():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    print("Training adaptive schema selector...")
    print()

    # For this tiny prototype dataset we use a stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Test accuracy:", round(accuracy, 4))
    print()

    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Feature importance:")

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print(importance.to_string(index=False))

    print()
    print("Predictions:")
    
    result = X_test.copy()
    result["actual"] = y_test.values
    result["predicted"] = predictions

    print(result.to_string(index=False))

    return model


if __name__ == "__main__":
    train_model()
