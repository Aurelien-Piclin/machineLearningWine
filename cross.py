import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np


df = pd.read_csv("winequality-white.csv", sep=";")

'''
# LA CROSS VAMIDATION EST ELLE UTILE ?
#melange pour une distribution equitable
df = df.sample(frac=1).reset_index(drop=True)

X = df.drop(columns='quality')
y = df['quality']

#import du modele le plus performant
decisionTree = tree.DecisionTreeClassifier()


#evaluation de la performance sur 10 sous groupe
scores = cross_val_score(decisionTree, X, y, cv=10)
print(scores)

#moyenne des scores
print(scores.mean())
#on remarque ici que la cross validation améliorera notre modele
'''


#On va donc diviser en 2
#Les données pour la Cross Validation, qu’on appellera train_test
#Les données pour tester les modèles finaux, qu’on appellera gtest pour global test
df = df.sample(frac=1).reset_index(drop=True)
X = df.drop(columns='quality')
y = df['quality']

X_train_test, X_gtest, y_train_test, y_gtest = train_test_split(X, y, test_size=0.10)

decisionTree = tree.DecisionTreeClassifier()

cv_results = cross_validate(decisionTree, X_train_test, y_train_test, cv=10, return_estimator=True)

#score des 10modeles entraines
print(cv_results['test_score'])

#moyenne
print(cv_results['test_score'].mean())

#parcourir chacun de nos modèles et calculer le score pour X_gtest et y_gtest
gtest_score = []
for i in range(len(cv_results['estimator'])):
  gtest_score.append(cv_results['estimator'][i].score(X_gtest, y_gtest))

print(sum(gtest_score) / len(gtest_score))


#prediction sur les resultats bruts
result = []
for i in range(len(cv_results['estimator'])):
  result.append(int(cv_results['estimator'][i].predict(X_gtest.iloc[:1])))

print(result)

#extrait la valeur la plus predite
print(max(set(result), key=result.count))

#compare a la valeur reelle
print(y_gtest.iloc[0])


#prediction pour les probabilités
#addition des probas
result_proba = cv_results['estimator'][0].predict_proba(X_gtest.iloc[:1])
for i in range(1, len(cv_results['estimator'])):
  result_proba =+ np.add(result_proba, cv_results['estimator'][i].predict_proba(X_gtest.iloc[:1]))

#moyenne des probas
result_proba = result_proba/10

#extraction de l'index avec la plus forte probabilité
np.argmax(result_proba)

wine_quality = [3, 4, 5, 6, 7, 8, 9]
print(wine_quality[np.argmax(result_proba)])