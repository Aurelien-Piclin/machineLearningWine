import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

#affichage arbre decision
import graphviz 

df = pd.read_csv("winequality-white.csv", sep=";")

#Dataframe feature
df_features = df.drop(columns='quality')

#DataFrame de notre cible
df_label = df['quality']

#séparation des données
X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, test_size=0.20)

#verif dimension des ensembles
'''
print((len(X_train), len(y_train)))
print((len(X_test), len(y_test)))
'''

#Logistic regression
'''
logisticRegression = LogisticRegression(max_iter=1000)
logisticRegression.fit(X_train, y_train)

print(logisticRegression.score(X_test, y_test))
'''
#visualisation des stats du premier vin (attention = notation en nombre scientifique)
'''
print(logisticRegression.predict_proba(X_test.iloc[:1]))
'''
#le résultat ciblé
'''
print(y_test.iloc[:1])
'''

#Support Vector Machines (SVM)
'''
SVM = svm.SVC()
SVM.fit(X_train, y_train)

print(SVM.score(X_test, y_test))
'''

#Stochastic Gradient Descent (SGD) par défaut il optimise un SVM
'''
SGD = SGDClassifier()
SGD.fit(X_train, y_train)

print(SGD.score(X_test, y_test))
'''

#Naive Bayes
'''
GNB = GaussianNB()
GNB.fit(X_train, y_train)

print(GNB.score(X_test, y_test))
'''

#Decision Tree
decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)

print(decisionTree.score(X_test, y_test))

#création PDF arbre décision
dot_data = tree.export_graphviz(decisionTree, out_file=None,
                                feature_names=df_features.columns.to_list(),
                                class_names=df_label.name,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("wine")
