#!/usr/bin/env python
# coding: utf-8

# # 1.import CSV

# In[1]:


import pandas as pd
data=pd.read_csv("food_coded.csv")


# # 2. Analysis of data accquired

# In[2]:


data


# In[3]:


data.columns


# In[4]:


column=['cook','eating_out','employment','ethnic_food', 'exercise','fruit_day','income','on_off_campus','pay_meal_out','sports','veggies_day']
d=data[column]
d


# # 3. Visualize

# In[5]:


import seaborn as sns
sns.pairplot(d)


# In[6]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
#% matplotlib inline 
ax=d.boxplot(figsize=(16,6))
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)


# In[7]:


d.shape


# In[8]:


s=d.dropna()


# # importing libraries

# In[9]:


## for data
import numpy as np
import pandas as pd
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for geospatial
import folium
import geopy
## for machine learning
from sklearn import preprocessing, cluster
import scipy
## for deep learning
import minisom


# # Elbow Method
# 

# In[10]:


f=['cook','income']
X = s[f]
max_k = 10
## iterations
distortions = [] 
for i in range(1, max_k+1):
    if len(X) >= i:
       model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
       model.fit(X)
       distortions.append(model.inertia_)
## best k: the lowest derivative
k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
     in np.diff(distortions,2)]))
## plot
fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color="red", label="k = "+str(k))
ax.set(title='The Elbow Method', xlabel='Number of clusters', 
       ylabel="Distortion")
ax.legend()
ax.grid(True)
plt.show()


# # REST API (For Apartments)

# In[11]:


from pandas import json_normalize
import folium
from geopy.geocoders import Nominatim 
import requests
CLIENT_ID = "PLtGnER5lyLJ2vtZq6dOlx7wj_bgNK3jCZfESSO675w" # your Foursquare ID
#CLIENT_SECRET = "KNG2LO22BPLHN1E3OAHWLYQ5PQBN14XYZMEMAS0CPJEJKOTR" # your Foursquare Secret 18.512497,73.781916
#VERSION = '20200316'
#LIMIT = 10000
circle=str(input('enter co-ordinates for POI: '))
url = ('https://discover.search.hereapi.com/v1/discover?in=circle:{};r=50000&q=apartment&apiKey=PLtGnER5lyLJ2vtZq6dOlx7wj_bgNK3jCZfESSO675w'.format(circle))
results = requests.get(url).json()
results


# In[12]:


venues = results['items']
nearby_venues = json_normalize(venues)


# In[13]:


nearby_venues


# # Cleaning API generated CSV

# In[14]:


#Cleaning API data
d2=nearby_venues[['title','address.label','distance','access','position.lat','position.lng','address.postalCode','contacts','id']]
d2.to_csv('cleaned_apartment.csv')


# In[15]:


#Counting no. of cafes, department stores and gyms near apartments around IIT Bombay
df_final=d2[['position.lat','position.lng']]


# # gathering additional feature data with REST API

# In[16]:


CafeList=[]
DepList=[]
GymList=[]
latitudes = list(d2['position.lat'])
longitudes = list( d2['position.lng'])
for lat, lng in zip(latitudes, longitudes):    
    radius = '5000' #Set the radius to 1000 metres
    latitude=lat
    longitude=lng
    
    search_query = 'cafe' #Search for any cafes
    url = 'https://discover.search.hereapi.com/v1/discover?in=circle:18.512497,73.781916;r=5000&q=Cafes&apiKey=uJHMEjeagmFGldXp661-pDMf4R-PxvWIu7I68UjYC5Q'.format(latitude, longitude, radius, search_query)
    results = requests.get(url).json()
    venues=json_normalize(results['items'])
    CafeList=venues["title"].count()
    
    search_query = 'gym' #Search for any cafes
    url = 'https://discover.search.hereapi.com/v1/discover?in=circle:18.512497,73.781916;r=5000&q=Gyms&apiKey=uJHMEjeagmFGldXp661-pDMf4R-PxvWIu7I68UjYC5Q'.format(latitude, longitude, radius, search_query)
    results = requests.get(url).json()
    venues=json_normalize(results['items'])
    GymList=venues["title"].count()
    
    search_query = 'gym' #Search for any cafes
    url = 'https://discover.search.hereapi.com/v1/discover?in=circle:18.512497,73.781916;r=5000&q=Grocery+stores&apiKey=uJHMEjeagmFGldXp661-pDMf4R-PxvWIu7I68UjYC5Q'.format(latitude, longitude, radius, search_query)
    results = requests.get(url).json()
    venues=json_normalize(results['items'])
    DepList=venues["title"].count()
#DepList


# In[17]:


from tabulate import tabulate
nearby_venues['Cafes'] = CafeList
nearby_venues['Department Stores'] = DepList
nearby_venues['Gyms'] = GymList

print(tabulate(nearby_venues,headers='keys',tablefmt='github'))


# # Elbow Method 

# In[18]:


f=['position.lat','position.lng']
X = nearby_venues[f]
max_k = 10
## iterations
distortions = [] 
for i in range(1, max_k+1):
    if len(X) >= i:
       model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
       model.fit(X)
       distortions.append(model.inertia_)
## best k: the lowest derivative
k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
     in np.diff(distortions,2)]))
## plot
fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color="red", label="k = "+str(k))
ax.set(title='The Elbow Method', xlabel='Number of clusters', 
       ylabel="Distortion")
ax.legend()
ax.grid(True)
plt.show()


# # Targetted point of Interest (PVPIT,Pune)

# In[19]:


city = "Pune"
## get location
locator = geopy.geocoders.Nominatim(user_agent="MyCoder")
location = locator.geocode(city)
print(location)
## keep latitude and longitude only
location = [location.latitude, location.longitude]
print("[lat, long]:", location)


# In[20]:


nearby_venues.head()


# In[21]:


nearby_venues.columns


# In[22]:


n=nearby_venues.drop(['id','language','resultType', 'access',],axis=1)


# In[23]:


n.columns


# In[24]:


n=nearby_venues
n


# In[25]:


nearby_venues['address.label']


# In[26]:


spec_chars = ["[","]"]
for char in spec_chars:
    nearby_venues['address.label'] = nearby_venues['address.label'].astype(str).str.replace(char, ' ')


# In[27]:


nearby_venues


# # generalized visual (map)
# 

# In[28]:


x, y = 'position.lat','position.lng'
color = "Cafes"
size = "Gyms"
popup = "address.label"
data = n.copy()

## create color column
lst_colors=["red","green","orange"]
lst_elements = sorted(list(n[color].unique()))

## create size column (scaled)
scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
data["size"] = scaler.fit_transform(
               data[size].values.reshape(-1,1)).reshape(-1)

## initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
## add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]],popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)
## add html legend


## plot the map
map_


# In[29]:


X = n[['position.lat','position.lng']]
max_k = 10
## iterations
distortions = [] 
for i in range(1, max_k+1):
    if len(X) >= i:
       model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
       model.fit(X)
       distortions.append(model.inertia_)
## best k: the lowest derivative
k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i in np.diff(distortions,2)]))
## plot
fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color="red", label="k = "+str(k))
ax.set(title='The Elbow Method', xlabel='Number of clusters', 
       ylabel="Distortion")
ax.legend()
ax.grid(True)
plt.show()


# In[30]:


k = 6
model = cluster.KMeans(n_clusters=k, init='k-means++')
X = n[['position.lat','position.lng']]
## clustering
dtf_X = X.copy()
dtf_X["cluster"] = model.fit_predict(X)
## find real centroids
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, 
                     dtf_X.drop("cluster", axis=1).values)
dtf_X["centroids"] = 0
for i in closest:
    dtf_X["centroids"].iloc[i] = 1
## add clustering info to the original dataset
n[["cluster","centroids"]] = dtf_X[["cluster","centroids"]]
n


# # Visualize KMeans Clusters

# In[31]:


## plot
fig, ax = plt.subplots()
sns.scatterplot(x='position.lat', y='position.lng', data=n, 
                palette=sns.color_palette("bright",k),
                hue='cluster', size="centroids", size_order=[1,0],
                legend="brief", ax=ax).set_title('Clustering (k='+str(k)+')')
th_centroids = model.cluster_centers_
ax.scatter(th_centroids[:,0], th_centroids[:,1], s=50, c='black', 
           marker="x")


# In[32]:


model = cluster.AffinityPropagation()


# In[33]:


k = n["cluster"].nunique()
sns.scatterplot(x="position.lat", y="position.lng", data=n, 
                palette=sns.color_palette("bright",k),
                hue='cluster', size="centroids", size_order=[1,0],
                legend="brief").set_title('Clustering (k='+str(k)+')')


# # Final Output

# In[34]:


x, y = "position.lat", "position.lng"
color = "cluster"
size = "Cafes"
popup = "address.label"
marker = "centroids"
data = n.copy()
## create color column
lst_elements = sorted(list(n[color].unique()))
lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in 
              range(len(lst_elements))]
data["color"] = data[color].apply(lambda x: 
                lst_colors[lst_elements.index(x)])
## create size column (scaled)
scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
data["size"] = scaler.fit_transform(
               data[size].values.reshape(-1,1)).reshape(-1)
## initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
## add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], 
           color=row["color"], fill=True,popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)
## add html legend
legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""
for i in lst_elements:
     legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
     fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
     </i>&nbsp;"""+str(i)+"""<br>"""
legend_html = legend_html+"""</div>"""
map_.get_root().html.add_child(folium.Element(legend_html))
## add centroids marker
lst_elements = sorted(list(n[marker].unique()))
data[data[marker]==1].apply(lambda row: 
           folium.Marker(location=[row[x],row[y]], 
           draggable=False,  popup=row[popup] ,
           icon=folium.Icon(color="black")).add_to(map_), axis=1)
## plot the map
locations = folium.map.FeatureGroup()
map_.add_child(locations)
folium.Marker([18.512497, 73.781916],popup='PVPIT Pune').add_to(map_)
map_


# # Save the map

# In[35]:


#saving the map 
map_.save("map-PVPIT_Pune_prep.html")


# In[ ]:




