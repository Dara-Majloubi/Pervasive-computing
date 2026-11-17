
# Define the machine learning classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def create_classifiers(random_state=42):
    classifiers = {}

    # Define the selected KNN classifier
    classifiers["KNN (k=3)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ])

    # Define the Decision Tree classifier
    classifiers["Decision Tree"] = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            random_state=random_state
        ))
    ])

    # Define the Multi-Layer Perceptron classifier
    classifiers["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=False,
            random_state=random_state
        ))
    ])

    return classifiers
