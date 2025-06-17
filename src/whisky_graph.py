import graphviz
import pandas as pd
import numpy as np
import networkx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import geopy.distance

# Read Whisky Regions Dataset
whisky_regions = pd.read_csv("../datasets/Whisky_Regions_Dataset.csv")
whisky_regions_features = whisky_regions[['Body','Sweetness','Smokey','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
whisky_regions_labels = whisky_regions[['Region']]

# Read Whisky Distileries Dataset
whisky_distileries = pd.read_csv("../datasets/whisky_distilieries.csv")
whisky_distileries_features = whisky_distileries[['Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
whisky_distileries_labels = whisky_distileries[['Distillery']]

partial_df = whisky_distileries[['Distillery','Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
partial_df.to_csv("Partial_Distilery_Data.csv")


# Whisky feature labels
feature_names = ['Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']

# y = np.ravel(y)
# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(whisky_regions_features, whisky_regions_labels)


# Get feature importances
importances = clf.feature_importances_


plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.show()
exit()

# from scipy import spatial

# d1_a = [2,3,1]
# d1_b = [1,0,1,1,1,0,1,1,2]
# d2_a = [2,1,2]
# d2_b = [1,0,1,1,0,2,1,1,1]

# result1 = 1 - spatial.distance.cosine(d1_a, d2_a)
# result2 = 1 - spatial.distance.cosine(d1_b, d2_b)

# r = result1 + result2 - 1

# print(r)