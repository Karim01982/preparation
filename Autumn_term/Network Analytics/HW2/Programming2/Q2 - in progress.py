# ==================================================== Importing Packages =========================================================== #

import plotly.plotly as py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
from gurobipy import *
from mpl_toolkits.basemap import Basemap


# ==================================================== 2(a) =========================================================== #
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
#print (dist_matrix)

# Creating Index for Dictionary 
index = list(range(0, n))

#Dictionary
cities_dict = {k: v for k, v in zip(index, x_y)}

lat = tsp_df['latitude'].values.tolist()
long = tsp_df['longitude'].values.tolist()

#Plotting only points
plt.figure(figsize=(12,12))
plt.scatter(long, lat)

#Plotting points on Map
m = Basemap(llcrnrlon=41.75, llcrnrlat=10.94, urcrnrlon=43.4, urcrnrlat=12.72, epsg=4713)
#m.drawcoastlines()
#m.drawstates()
#m.drawcountries()

m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= False)
#m.arcgisimage(service='NatGeo_World_Map', xpixels = 4000, verbose= False, interpolation='bicubic')
#m.arcgisimage(server='http://maps.ngdc.noaa.gov',service='etopo1',verbose=True)
#m.arcgisimage(service='World_Shaded_Relief', xpixels = 2000, verbose= False)
#m.arcgisimage(service='World_Physical_Map', xpixels = 2000, verbose= False)
#m.arcgisimage(service='World_Imagery', xpixels = 2000, verbose= False)
#m.shadedrelief()
#m.bluemarble()
#m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

#m.drawmapboundary(fill_color='aqua')
## fill continents, set lake color same as ocean color.
#m.fillcontinents(color='coral',lake_color='aqua')

#m.drawcoastlines()
#m.drawstates()
#m.drawcountries()

for i in list(range(n)):
    x, y = m(float(long[i]), float(lat[i]))
    m.plot(x, y, marker = 'o', markeredgecolor='black', color = 'red')
plt.figure(figsize=(12,12))
plt.show()
#
#m1 = Basemap(width=200000,height=220000,projection='lcc', resolution='h',lat_0=11.824,lon_0=42.596)
#m1.drawcoastlines()
#m1.drawcountries()
#m1.bluemarble()
##m.arcgisimage(service='NatGeo_World_Map', xpixels = 4000, verbose= False, interpolation='bicubic')
#m1.scatter(x,y,10,marker='o', color='b')
#plt.show()