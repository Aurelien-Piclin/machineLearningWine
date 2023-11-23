import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv("winequality-white.csv", sep=";")
features = df.drop(['quality'], axis=1)
labels = df[['quality']]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


decisionTree = tree.DecisionTreeClassifier(criterion="entropy",
                                           class_weight={3:1, 4:2, 5:4, 6:4, 7:3, 8:2, 9:1}
                                           )

#entrainement de l'arbre
decisionTree.fit(X_train, y_train)

#calcul performance
print(decisionTree.score(X_test, y_test))