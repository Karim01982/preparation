# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 01:53:52 2017

@author: 64254
"""

#2
#[10 or 15 points] The data HW2_tsp.txt contains latitude and longitude 
#data of 38 cities in a country in Africa (Djibouti). Calculate the distance
#matrix (use an approximation like here which is quite accurate for short 
#distances; or use packages like haversine or geopy). The x and y-cordinates
#are the latitude and longitude in decimal form multiplied by 1000.
#EUC_2D means take these as Cartesian co-ordinates. Can also use haversine 
#treating them as longitude and latitude. You are free to use any other functions
#you find.


import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import math
from mpl_toolkits.basemap import Basemap

# the original csv file already delete unnecessary comment.
# read the txt file
tsp = pd.read_csv('HW2_tsp1.csv', index_col = False)

#split the column into 3 columns

foo = lambda x: pd.Series([i for i in (x.split(' '))])
tsp = tsp['node'].apply(foo)
tsp.rename(columns={0:'city',1:'latitude',2:'longitude'},inplace=True)
tsp[['city','latitude','longitude']]
tsp.info()



#convert the longitude and altitude into float value
tsp['latitude'] = pd.to_numeric(tsp['latitude'].str.replace(' ',''), errors='force')
tsp['latitude'] = tsp['latitude']/1000
tsp['longitude'] = pd.to_numeric(tsp['longitude'].str.replace(' ',''), errors='force')
tsp['longitude'] = tsp['longitude']/1000

tsp[tsp.columns[1]].iloc[1] # the value of first column and 1 row.
tsp.loc[tsp.index[0], tsp.columns[0]]  # the value of first column and 1 row.
tsp.info()
tsp.loc[0][tsp.columns[0]]
tsp.index
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    
 
# Convert latitude and longitude to
# spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
 
# phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
 
# theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
 
# Compute spherical distance from spherical coordinates.
 
# For two locations in spherical coordinates
# (1, theta, phi) and (1, theta', phi')
# cosine( arc length ) =
# sin phi sin phi' cos(theta-theta') + cos phi cos phi'
# distance = rho * arc length
 
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
    math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    distance_in_km = arc * 6373
 
# Remember to multiply arc by the radius of the earth
# in your favorite set of units to get length.
    return distance_in_km

tsp_dist = np.zeros(shape = (38, 38))
for i in list(range(38)):
    for j in list(range(38)):
        if i != j:
            tsp_dist[i, j] = distance_on_unit_sphere(tsp.iloc[i, 1], tsp.iloc[i, 2], tsp.iloc[j, 1], tsp.iloc[j, 2])

"""
a.Plot the latitude and longitude as a scatter plot using a drawing package 
(some options are: matplotlib basemap toolkit (the most advanced, but also the
most difficult to use), geopy, gmplot, plotly …).
"""
fig = plt.figure(figsize = (12,12))
m = Basemap(llcrnrlon=41.5,
            llcrnrlat=10.5,
            urcrnrlon=43.8,
            urcrnrlat=13, epsg=4713)
#http://server.arcgisonline.com/arcgis/rest/services
#EPSG Number of America is 4269
m.arcgisimage(service='ESRI_Imagery_World_2D', 
              xpixels = 2000, 
              verbose= True)

# Map (long, lat) to (x, y) for plotting
for i in tsp.index:
    x, y = m(tsp.iloc[i,2], tsp.iloc[i,1])
    m.plot(x,y,marker='o',color='red')
    
parallels = np.arange(0,40, 0.5)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,True,True,True])
meridians = np.arange(10,50, 0.5)
m.drawmeridians(meridians,labels=[True,True,True,True])
plt.show() ###why setting resolution= f still vague?  

"""
b. (5 points) Use the or-tools traveling salesman routine to find a tour of the
 38 cities. If it is not finding a tour fast enough, try a subset of 10 cities 
 (then 15 and so on). My suggestion is to try to understand the solved example
 given there and try to recreate it for this data.

"""

import random
import argparse
from ortools.constraint_solver import pywrapcp
# You need to import routing_enums_pb2 after pywrapcp!
from ortools.constraint_solver import routing_enums_pb2

list2 = list() # a list to store the node in the tour
parser = argparse.ArgumentParser()
parser.add_argument('--tsp_size', default = 38, type = int,
                     help='Size of Traveling Salesman Problem instance.')
parser.add_argument('--tsp_use_random_matrix', default=True, type=bool,
                     help='Use random cost matrix.')
parser.add_argument('--tsp_random_forbidden_connections', default = 0,
                    type = int, help='Number of random forbidden connections.')
parser.add_argument('--tsp_random_seed', default = 0, type = int,
                    help = 'Random seed.')
parser.add_argument('--light_propagation', default = False,
                    type = bool, help = 'Use light propagation')
#Create an empty dataframe

def Distance(i, j):
  """Sample function."""
  # Put your distance code here.
  return i + j


class RandomMatrix(object):
  """Random matrix."""

  def __init__(self, size, seed):
    """Initialize random matrix."""

    self.matrix = tsp_dist
   
  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


def main(args):
  # Create routing model
  if args.tsp_size > 0:
    # TSP of size args.tsp_size
    # Second argument = 1 to build a single tour (it's a TSP).
    # Nodes are indexed from 0 to parser_tsp_size - 1, by default the start of
    # the route is node 0.
    routing = pywrapcp.RoutingModel(args.tsp_size, 1, 0)

    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    # Setting first solution heuristic (cheapest addition).
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Setting the cost function.
    # Put a callback to the distance accessor here. The callback takes two
    # arguments (the from and to node inidices) and returns the distance between
    # these nodes.
    matrix = RandomMatrix(args.tsp_size, args.tsp_random_seed)
    matrix_callback = matrix.Distance
    if args.tsp_use_random_matrix:
      routing.SetArcCostEvaluatorOfAllVehicles(matrix_callback)
    else:
      routing.SetArcCostEvaluatorOfAllVehicles(Distance)
    # Forbid node connections (randomly).
    rand = random.Random()
    rand.seed(args.tsp_random_seed)
    forbidden_connections = 0
    while forbidden_connections < args.tsp_random_forbidden_connections:
      from_node = rand.randrange(args.tsp_size - 1)
      to_node = rand.randrange(args.tsp_size - 1) + 1
      if routing.NextVar(from_node).Contains(to_node):
        print('Forbidding connection ' + str(from_node) + ' -> ' + str(to_node))
        routing.NextVar(from_node).RemoveValue(to_node)
        forbidden_connections += 1

    # Solve, returns a solution if any.
#    assignment = routing.SolveWithParameters(search_parameters)
    assignment = routing.Solve()
    if assignment:
      # Solution cost.
      print(assignment.ObjectiveValue())
      # Inspect solution.
      # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
      route_number = 0
      node = routing.Start(route_number)
      route = ''
      while not routing.IsEnd(node):
        route += str(node) + ' -> '
        list2.append(str(node))# append the node into a list
        node = assignment.Value(routing.NextVar(node))
      route += '0'
      print(route)    
    else:
      print('No solution found.')
  else:
    print('Specify an instance greater than 0.')

if __name__ == '__main__':
  main(parser.parse_args())

columns = ['City1_index', 'City2_index']
tour = pd.DataFrame(index = range(38), columns=columns)
for i in range(37):
    tour['City1_index'].iloc[i] = str(list2[i])
    tour['City2_index'].iloc[i] = str(list2[i + 1])
tour['City1_index'].iloc[37] = str(list2[37])
tour['City2_index'].iloc[37] = str(list2[0])
tour

"""
d. Plot the resulting tour on the scatter plot
"""
fig = plt.figure(figsize = (12,12))
m = Basemap(llcrnrlon=41.5,
            llcrnrlat=10.5,
            urcrnrlon=43.8,
            urcrnrlat=13, epsg=4713)
#http://server.arcgisonline.com/arcgis/rest/services
#EPSG Number of America is 4269
m.arcgisimage(service='ESRI_Imagery_World_2D', 
              xpixels = 2000, 
              verbose= True)
#create a dataframe to contain each dots and its latitude and longitude
pos={}
tsp.iloc[0,1]
for i in range(len(list2)):
    pos[list2[i]] = (tsp.iloc[i,2], tsp.iloc[i,1])

# The NetworkX part
# put map projection coordinates in pos dictionary
G=nx.Graph()
for i in range(len(tour.index)):
    G.add_edge(tour.iloc[i, 0], tour.iloc[i,1])

# draw
nx.draw_networkx(G,pos,node_size=200,node_color='pink')
parallels = np.arange(0,40, 0.5)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,True,True,True])
meridians = np.arange(10,50, 0.5)
m.drawmeridians(meridians,labels=[True,True,True,True])
#plt.title('Tour in Djibouti')
plt.show() 
