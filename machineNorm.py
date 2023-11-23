import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import tree

df = pd.read_csv("winequality-white.csv", sep=";")

#Dataframe feature
df_features = df.drop(columns='quality')

#DataFrame de notre cible
df_label = df['quality']


#obtenir le min, max, moyene et ecart typpe d'une colonne
print(f'Min : ',df_features['fixed acidity'].min(),', Max :', df_features['fixed acidity'].max())
print(f'Mean : ',round(df_features['fixed acidity'].mean(),2),', Standard Deviation :', round(df_features['fixed acidity'].std(),2))

'''
EXEMPLE
#Normalisation [0,1]
#initialisation MinMaxScaler
transformer = preprocessing.MinMaxScaler().fit(df_features[['fixed acidity']])

#transformation
X_transformed = transformer.transform(df_features[['fixed acidity']])

#Pour info: pour repasser aux valeurs précédentes il faut utiliser la fonction inverse_transform

#ATTENTION => la transformation convertit vos DataFrame en Numpy Array. Pour avoir un DataFrame au lieu d’un Numpy Array, utilisez après l’opération de normalisation : df = pd.DataFrame(X_transformed, columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']).


#Normalisation [-1,1]
transformer2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(df_features[['fixed acidity']])
X_transformed2 = transformer2.transform(df_features[['fixed acidity']])


#Loi normale 
transformer = preprocessing.StandardScaler().fit(df_features[['fixed acidity']])
X_transformed3 = transformer.transform(df_features[['fixed acidity']])

'''

#initialisation sur toutes les features
transformer = preprocessing.StandardScaler().fit(df_features)

#transformation
df_features_transformed = transformer.transform(df_features)

#separation données
X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, test_size=0.20)

#decision tree
decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)

print(decisionTree.score(X_test, y_test))