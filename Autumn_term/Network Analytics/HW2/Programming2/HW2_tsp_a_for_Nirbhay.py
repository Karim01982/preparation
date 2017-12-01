#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:24:29 2017

@author: Veronique
"""

# The data HW2_tsp.txt contains latitude and longitude data of 38 cities in a country in Africa (Djibouti).
# Calculate the distance matrix (use an approximation like here which is quite accurate for short distances; 
# or use packages like haversine or geopy).
# The x and y-cordinates are the latitude and longitude in decimal form multiplied by 1000.
# EUC_2D means take these as Cartesian co-ordinates.
# Can also use haversine treating them as longitude and latitude.
# You are free to use any other functions you find.

from pandas import Series, DataFrame
import pandas as pd
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import math

tsp_raw = pd.read_csv('/Users/Veronique/Desktop/ICBA/Nov_Dec Term/Network Analytics/Homework/Hw2/HW2_tsp_clean.csv')

# calculate distance matrix
def distance_km(lat1, long1, lat2, long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians
 
    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians
 
    # Compute spherical distance from spherical coordinates.
    
    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    
    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
           math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)
    return arc * 6373

tsp_dist = np.zeros(shape = (38, 38))
for i in list(range(38)):
    for j in list(range(38)):
        if i != j:
            tsp_dist[i, j] = distance_km(tsp_raw.iloc[i, 1], tsp_raw.iloc[i, 2], tsp_raw.iloc[j, 1], tsp_raw.iloc[j, 2])

# a. Plot the latitude and longitude as a scatter plot using a drawing package 
# (some options are: matplotlib basemap toolkit (the most advanced, but also the most difficult to use), geopy, gmplot, plotly …).  
 # Cities names and coordinates
cities = [str(i) for i in list(range(38))]
lat = list(float(tsp_raw.iloc[:, 1]))
lon = list(float(tsp_raw.iloc[:, 2]))

# setup basemap.
m = Basemap(width = 360000, 
            height = 240000,
            projection = 'lcc',
            resolution = 'f',
            lat_0 = 11.8,
            lon_0 = 42.5)
m.bluemarble()
# plot cities
for i in list(range(38)):
    x, y = m(float(lon[i]), float(lat[i]))
    m.plot(x, y, marker = 'v', color = 'coral')
plt.show()
plt.savefig('City Scatter Plot')


# b. (5 points) Use the or-tools traveling salesman routine to find a tour of the 38 cities.  
# If it is not finding a tour fast enough, try a subset of 10 cities (then 15 and so on).  
# My suggestion is to try to understand the solved example given there and try to recreate it for this data.  




# c. [Challenge problem., 10 points]  The most powerful integer programming solver is Gurobi (along with CPLEX).  
# They give a free one year license if you download and install from a University IP address.  
# Use the most powerful computer you have in your group (cores and memory).  Connect with the Python interface of Gurobi and find an optimal tour using the sub-tour elimination integer programming formulation we did in class.  
# You cannot generate all the sub-tour elimination constraints at once (too many). 
# Instead,  i. Start with a problem with only the degree =2 constraints.   
# ii. Check the resulting solution for sub-tours.  If there is a sub-tour, add the sub-tour elimination constraints corresponding to a cut defined by the sub-tour (i.e. you are eliminating that sub-tour in the next round).   
# iii. Repeat till you find a tour (in which case, it is optimal).    Make sure you do not discard the solution from the previous round, but start from what you had found.  
# This method of generating constraints on the fly is suitable when the number of constraints are exponential in the problem size.  
# The dual version of this is called column generation.  
# Compare the optimal solution with the or-tools solution.  The optimal tour is here.  






# d. Plot the resulting tour on the scatter plot 




