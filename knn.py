from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
print("Iris Dataset Loaded ----")
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print("\nComplete Iris Dataset:")
print(iris_df.to_string(index=False))
print("\nTarget Labels:")
for i in range(len(iris.target_names)):
    print("Label", i, ":", iris.target_names[i])
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print("\nPredictions:")
correct = 0
incorrect = 0
for i in range(len(X_test)):
    actual_label = Y_test[i]
    predicted_label = Y_pred[i]
    status = "Correct" if actual_label == predicted_label else "Wrong"
    if status == "Correct":
        correct += 1
    else:
        incorrect += 1
    print(f"Sample: {X_test[i]}, Actual: {iris.target_names[actual_label]}, "
          f"Predicted: {iris.target_names[predicted_label]} - ({status})")
print("\nSummary:")
print("Total samples:", len(Y_test))
print("Correct predictions:", correct)
print("Wrong predictions:", incorrect)
print("Accuracy:", classifier.score(X_test, Y_test))
