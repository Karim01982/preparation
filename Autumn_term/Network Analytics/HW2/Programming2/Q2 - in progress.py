import plotly.plotly as py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
from gurobipy import *
from mpl_toolkits.basemap import Basemap

raw_data = r'C:/Users/Nirbhay Sharma/Coursework/02-Autumn Term/05-Nework Analytics/Homework/2/HW2_tsp.txt'
tsp_df = pd.read_table(raw_data, delim_whitespace=True, names=('latitude', 'longitude'), skiprows = (0,1,2,3,4,5,6,7,8,9))

tsp_df['latitude']/=1000
tsp_df['longitude']/=1000

tsp_array = np.array(tsp_df)
x_y = tsp_df.values.tolist()

n = tsp_array.shape[0]
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = haversine(tsp_array[i], tsp_array[j])
        dist_matrix[j, i] = dist_matrix[i, j]

#Printing the distance matrix
print (dist_matrix)

# Creating Index for Dictionary 
index = list(range(0, n))

#Dictionary
cities_dict = {k: v for k, v in zip(index, x_y)}

#Part a
lat = tsp_df['latitude'].values.tolist()
long = tsp_df['longitude'].values.tolist()
plt.figure(figsize=(12,12))
plt.scatter(long, lat)

#Part 2(a) better display
m = Basemap(llcrnrlon=41.75, llcrnrlat=10.94, urcrnrlon=43.4, urcrnrlat=12.72, epsg=4713)
m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= False)
#m.arcgisimage(service='World_Shaded_Relief', xpixels = 8000, verbose= False)
for i in list(range(n)):
    x, y = m(float(long[i]), float(lat[i]))
    m.plot(x, y, marker = 'o', color = 'red')
plt.figure(figsize=(18,18))
plt.show()

