#https://inside-machinelearning.com/scikit-learn-projet-machine-learning/

import sys
import pandas as pd


# deux librairies principales pour tracer des graphiques
import seaborn as sns
import matplotlib.pyplot as plt

#graphique en bar
import numpy as np

# DataFrame (ou objet pandas)
df = pd.read_csv("winequality-white.csv", sep=";")

#afficher le type de données de chaque colonne
# print(df.dtypes)

#afficher les 3 premieres lignes
'''
print(df.head(3))
'''

#si potentiel infini ie qu'il y a une liste tres grande => modele de regression
#si potentiel restreint ie qu'il y a une petite liste (c'est le cas ici) => modèle de classification
#
'''
print(df['quality'].unique())
'''

#creation de graphique, on remarque une distribution non égale (peu de vin qualité 9)
"""
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
df['quality'].value_counts(normalize=True).plot.bar(rot=0, color='#066b8b')
plt.ylabel('quality')
plt.xlabel('% distribution per category')
plt.subplot(1,2,2)
sns.countplot(data=df,y='quality')
plt.tight_layout()
plt.show() 
"""

#nouveau dateFrame contenant uniquement les caracteristiques du vin
df_features = df.drop(columns='quality')


# [GLOBAL] on cherche ici a savoir avec ces 2 graphiques si il n'y a pas d'anormalité ie une distribution inhabituelle, des données manquantes ou un cas isolé
"""
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
ax = sns.kdeplot(df_features['alcohol'],fill=True,color='#d1aa00')
plt.ylabel('alcohol')
plt.xlabel('% distribution per category')
plt.subplot(1,2,2)
df_features['alcohol'].plot.box()
plt.tight_layout()
plt.show()
"""

#(CHAQUE COLONNE)
num_columns = df_features.columns.tolist()
num_columns

"""
plt.figure(figsize=(18,40))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(df[col],color='#d1aa00',fill=True)
    plt.subplot(8,4,i+11)
    df[col].plot.box()
plt.tight_layout() 
plt.show()
"""

#on utilise la Skewness et le Kurtosis pour évaluer la distribution de données https://inside-machinelearning.com/skewness-et-kurtosis/
'''
pd.DataFrame(data=[df[num_columns].skew(),df[num_columns].kurtosis()],index=['skewness','kurtosis'])
'''

#graphique en violon lien entre quantité de sulfate et qualité du vin 
'''
plt.figure(figsize=(16,6))
sns.violinplot(data=df, x='quality', y='sulphates')
'''

#graphique en nuage de point total dioxyde de souffre et qualité du vin => on remarque ici un biais de notre dataset car peu de donnée pour qualité 3 et 9
'''
plt.figure(figsize=(16,6))
sns.stripplot(x="quality", y="total sulfur dioxide", data=df, size=4)
# sns.swarmplot(x="quality", y="total sulfur dioxide", data=df)
'''

#calcul de la moyenne de dioxyde de souffre par qualité de vin
'''
quality_cat = df.quality.unique()
quality_cat.sort()
qual_TSD = []
for i,quality in enumerate(quality_cat):
  qual_TSD.append([quality, df['total sulfur dioxide'].loc[df['quality'] == quality].mean()])
'''
  
#on met ces données dans un nouveau dataFrame
'''
df_qual_TSD = pd.DataFrame(qual_TSD, columns =['Quality', 'Mean TSD'])
'''

#graphique
'''
plt.figure(figsize=(10,5))
sns.barplot(x="Quality", y="Mean TSD", data=df_qual_TSD)
plt.show()
'''


#transformer nos données numérique en donnée categorique
'''
def alcohol_cat(alcohol):
    if alcohol <= 9.5:
        return "Low"
    elif alcohol <= 11:
        return "Moderate"
    elif alcohol <= 12.5:
        return "High"
    else:
        return "Very High"

df['alcohol_category'] = df['alcohol'].apply(alcohol_cat)
print(df.sample(frac=1).head())
'''
'''
plt.figure(figsize=(15,30))

cross = pd.crosstab(index=df['quality'],columns=df['alcohol_category'],normalize='index')
cross.plot.barh(stacked=True,rot=40,cmap='crest_r').legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('% distribution per category')
plt.xticks(np.arange(0,1.1,0.1))
plt.title("Wine Quality each {}".format('alcohol_category'))
plt.show()
'''

#heatmap feature et quality
'''
plt.figure(figsize=(15,2))
sns.heatmap(df.corr().iloc[[-1]],
            cmap='RdBu_r',
            annot=True,
            vmin=-1, vmax=1)
plt.show()
'''

#heatmap entre toutes les données
'''
plt.figure(figsize=(15,2))
sns.heatmap(df.corr(),
            cmap='RdBu_r',
            annot=True,
            vmin=-1, vmax=1)
plt.show()
'''

sys.exit();