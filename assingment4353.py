# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:27:18 2018

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')

# since i have no dataset given i have simulated the data 

product=["topwear"]
color =["red", "blue" , "green " ]
spec=["t-shirt","shirt"]
brand_name=["nike","colt","ruggurs"]
price=[1000,700,2000,1550,1300]
size_of_sample = 100


col1=np.random.choice(product, size_of_sample)
col2=np.random.choice(color, size_of_sample)
col3=np.random.choice(spec,size_of_sample)
col4=np.random.choice(brand_name,size_of_sample)
col5=np.random.choice(price,size_of_sample)
my_dict={"prodcut name":col1,"color":col2,"type":col3,"brand":col4,"price":col5}

df=pd.DataFrame(my_dict)
X = df.iloc[:, :].values



#dealing with catogirical data


df=pd.DataFrame(my_dict)
X = df.iloc[:, :].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])


onehotencoder=OneHotEncoder(categorical_features = [0,1,3,4]);
X = onehotencoder.fit_transform(X).toarray()
#standerdizing the data

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X= sc_X.fit_transform(X)

# choosing the cluster points

wcss = []

for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Clustering the data
k = 10
kmeans = KMeans(n_clusters = k)
y_kmeans = kmeans.fit_predict(X)

labels = [('Cluster ' + str(i+1)) for i in range(k)]

# Plotting the clusters
plt.figure()
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 20,
                 c = cmap(i/k), label = labels[i]) 
 
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'black', label = 'Centroids', marker = 'X')

plt.title('Kmeans cluster plot')
plt.legend()
plt.show()

# getting the index of data of cluster it belongs to , so that when user inputs data
#we can fetch data from dataframe and output similar products as we know the cluster it belongs to


cluster1= [i for i,j in enumerate(y_kmeans) if j==0]
cluster2= [i for i,j in enumerate(y_kmeans) if j==1]
cluster3= [i for i,j in enumerate(y_kmeans) if j==2]
cluster4= [i for i,j in enumerate(y_kmeans) if j==3]
cluster5= [i for i,j in enumerate(y_kmeans) if j==4]
cluster6= [i for i,j in enumerate(y_kmeans) if j==5]
cluster7= [i for i,j in enumerate(y_kmeans) if j==6]
cluster8= [i for i,j in enumerate(y_kmeans) if j==7]
cluster9= [i for i,j in enumerate(y_kmeans) if j==8]
cluster10= [i for i,j in enumerate(y_kmeans) if j==9]

    
# user inputs data
        
print("Enter product type(available:topwear): ")
product1=input()
print("Enter color (available color: red, blue, green): ")
color1=input()
print("Enter price (range:700-2000): ")
price1=input()
print("Enter type (available:shirt,t_shirt):")
type1=input()
print("Enter the brand(availbe brands:nike,colt,ruggurs):")
brand1=input()



user_input=[["nike" ,"red" ,700 ,"topwear", "shirt"]]
labelencoder_X = LabelEncoder()
user_input[:, 0] = labelencoder_X.fit(user_input[:, 0])
user_input[:, 1] = labelencoder_X.fit(user_input[:, 1])
user_input[:, 3] = labelencoder_X.fit(user_input[:, 3])
user_input[:, 4] = labelencoder_X.fit(user_input[:, 4])


onehotencoder=OneHotEncoder(categorical_features = [0,1,3,4]);
user_input = onehotencoder.transform(user_input).toarray()

result = kmeans.predict(user_input)
""" pland was to know which cluster user belongs to ""
'"" i failed to convert user input to label encoding """




