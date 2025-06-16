import graphviz
import pandas as pd
import numpy as np
import networkx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import geopy.distance

df = pd.read_csv("../datasets/whisky_distilieries.csv")

partial_df = df[['Distillery','Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
partial_df.to_csv("Partial_Data.csv")

df_features = df[['Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
df_labels = df[['Distillery']]

names = ['Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']


# Load dataset
X = df_features
y = df_labels
feature_names = df_labels

y = np.ravel(y)
# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)


# Get feature importances
importances = clf.feature_importances_


plt.figure(figsize=(10, 6))
plt.bar(names, importances)
plt.show()

# from scipy import spatial

# d1_a = [2,3,1]
# d1_b = [1,0,1,1,1,0,1,1,2]
# d2_a = [2,1,2]
# d2_b = [1,0,1,1,0,2,1,1,1]

# result1 = 1 - spatial.distance.cosine(d1_a, d2_a)
# result2 = 1 - spatial.distance.cosine(d1_b, d2_b)

# r = result1 + result2 - 1

# print(r)