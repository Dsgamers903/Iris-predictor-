from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
import joblib

iris = load_iris()
x = iris.data  # type: ignore
y = iris.target  # type: ignore

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])


pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Your test accuracy: {accuracy}")

joblib.dump(pipeline, "model.pkl")
print("Saved trained pipeline to model.pkl")
