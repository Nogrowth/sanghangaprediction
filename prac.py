from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("train set accuracy: {:.3f}".format(tree.score(X_train, y_train)))
print("test set accuracy: {:.3f}".format(tree.score(X_test, y_test)))

import matplotlib.pyplot as plt
for idx, feature in enumerate(cancer.feature_names):
    for i, value in enumerate(y_test):
        if value == 0:


plt.figure(figsize=(30, 30))



