import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("winequality-white.csv", sep=";")


#On va donc diviser en 2
#Les données pour la Cross Validation, qu’on appellera train_test
#Les données pour tester les modèles finaux, qu’on appellera gtest pour global test

X = df.drop(columns='quality')
y = df['quality']

X_train_test, X_gtest, y_train_test, y_gtest = train_test_split(X, y, test_size=0.10)

decisionTree = tree.DecisionTreeClassifier(random_state=42)

cv = StratifiedKFold(n_splits=10, shuffle=True)

cv_results = cross_validate(decisionTree, X_train_test, y_train_test, cv=cv, return_estimator=True)

#score des 10modeles entraines
test_scores = cv_results['test_score']
print(test_scores)

#moyenne
print(test_scores.mean())

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

#moyenne pondéré en fonction des score de test
result_proba = np.zeros_like(cv_results['estimator'][0].predict_proba(X_gtest.iloc[:1]))
for i, estimator in enumerate(cv_results['estimator']):
    result_proba += estimator.predict_proba(X_gtest.iloc[:1]) * test_scores[i]

#extraction de l'index avec la plus forte probabilité
max_proba_index = np.argmax(result_proba)

wine_quality = [3, 4, 5, 6, 7, 8, 9]
predicted_quality = wine_quality[max_proba_index]
print(predicted_quality)