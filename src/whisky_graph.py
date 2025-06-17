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
whisky_regions_reduced =  whisky_regions[['Region','Body','Sweetness','Smokey','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]

# Read Whisky Distileries Dataset
whisky_distileries = pd.read_csv("../datasets/whisky_distilieries.csv")
whisky_distileries_features = whisky_distileries[['Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
whisky_distileries_labels = whisky_distileries[['Distillery']]

partial_df = whisky_distileries[['Distillery','Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']]
partial_df.to_csv("Partial_Distilery_Data.csv")


# Whisky feature labels
feature_names = ['Body','Sweetness','Smoky','Medicinal','Tobacco','Honey','Spicy','Winey','Nutty','Malty','Fruity','Floral']



from scipy import spatial

lowlands = whisky_regions_reduced.loc[whisky_regions_reduced['Region'] == 'Lowlands']
highlands = whisky_regions_reduced.loc[whisky_regions_reduced['Region'] == 'Highlands']
islay = whisky_regions_reduced.loc[whisky_regions_reduced['Region'] == 'Islay']
campbeltown = whisky_regions_reduced.loc[whisky_regions_reduced['Region'] == 'Campbeltown']
speyside = whisky_regions_reduced.loc[whisky_regions_reduced['Region'] == 'Speyside']

speyside = whisky_regions_reduced.loc[0]
speyside_list = list(speyside)
speyside_list = speyside_list[1:]

highlands = whisky_regions_reduced.loc[1]
highlands_list = list(highlands)
highlands_list = highlands_list[1:]

lowlands = whisky_regions_reduced.loc[2]
lowlands_list = list(lowlands)
lowlands_list = lowlands_list[1:]

islay = whisky_regions_reduced.loc[3]
islay_list = list(islay)
islay_list = islay_list[1:]

campbeltown = whisky_regions_reduced.loc[4]
campbeltown_list = list(campbeltown)
campbeltown_list = campbeltown_list[1:]

whisky_region_values_list = [speyside_list,highlands_list,lowlands_list,islay_list,campbeltown_list]
region_labels_list = whisky_regions_labels['Region'].tolist()


results = []
full_results = [[]]
i = 0

for whisky_list_a in whisky_region_values_list:
    
    # results.append(region_labels_list[i])
    for whisky_list_b in whisky_region_values_list:
        result = 1 - spatial.distance.cosine(whisky_list_a, whisky_list_b)
        # results.append(result)
        results.append(result)
    # full_results.append(results)    
    i +=1

normalizedData = (results-np.min(results))/(np.max(results)-np.min(results))    
print(normalizedData)




# # y = np.ravel(y)
# # Train a RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=1000)
# clf.fit(whisky_regions_features, whisky_regions_labels)


# # Get feature importances
# importances = clf.feature_importances_


# plt.figure(figsize=(10, 6))
# plt.bar(feature_names, importances)
# plt.show()
# exit()
