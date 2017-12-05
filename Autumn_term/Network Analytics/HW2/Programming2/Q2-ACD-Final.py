# ==================================================== 2(a) ==================================================== #

# IMPORTING THE PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
from gurobipy import *
from mpl_toolkits.basemap import Basemap
import networkx as nx

# CREATING THE CITIES DATAFRAME
# storing the path of the data file
raw_data = r'C:/Users/Nirbhay Sharma/Coursework/02-Autumn Term/05-Nework Analytics/Homework/2/HW2_tsp.txt'
# creating the dataframe
dj_tsp_df = pd.read_table(raw_data, header=None, skiprows=10, sep=" ")
# renaming the columns, and dropping the no.
dj_tsp_df.columns = ['no.','latitude','longitude']
dj_tsp_df = dj_tsp_df.drop(columns=['no.'])
dj_tsp_df = dj_tsp_df[['longitude','latitude']]
# dividing coordinates by 1,000
dj_tsp_df = dj_tsp_df[['longitude','latitude']]/1000

# CREATING NUMPY ARRAY WITH DISTANCES BETWEEN CITIES
# creating an empty numpy array
dj_dist_matrix = np.zeros((len(dj_tsp_df),len(dj_tsp_df)))
# populating the array by calculating distances between provided coordinates using Haversine function
for i in range(len(dj_tsp_df)):
    for j in range(len(dj_tsp_df)):
        dj_dist_matrix[i,j] = haversine((dj_tsp_df['latitude'][i], dj_tsp_df['longitude'][i]) , (dj_tsp_df['latitude'][j],dj_tsp_df['longitude'][j]))

# CREATING VARIABLES THAT STORE LATITUDE AND LONGITUDE VALUES
lat = dj_tsp_df['latitude'].values.tolist()
long = dj_tsp_df['longitude'].values.tolist()

# PLOTTING THE POINTS ON GRAPH
plt.figure(figsize=(12,12))
plt.scatter(long, lat, s=50, color='red')

# PLOTTING THE POINTS ON THE MAP
# creating variables for latitude and longitude lines
parallels = np.arange(0,40, 0.5)
meridians = np.arange(10,50, 0.5)
# creating the map using an image service
plt.figure(figsize=(12,12))
dj_map = Basemap(llcrnrlon=41.75, llcrnrlat=10.94, urcrnrlon=43.4, urcrnrlat=12.72, epsg=4713)
dj_map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= False)
# plotting the points
for i in list(range(len(dj_tsp_df))):
    x, y = dj_map(float(long[i]), float(lat[i]))
    dj_map.plot(x, y, marker = 'o', markeredgecolor='black', markersize=10, color = 'red')
# drawing latitude and longitude lines
dj_map.drawparallels(parallels,labels=[True,True,True,True])
dj_map.drawmeridians(meridians,labels=[True,True,True,True])
plt.show()

# ==================================================== 2(c) ==================================================== #

# ==================== Part (i) ====================
# defining variables for use
n = len(dj_tsp_df)
lat_long = dj_tsp_df.values.tolist()

# CREATING CITIES DICTIONARY
index = list(range(0, n))
cities_dict = {k: v for k, v in zip(index, lat_long)}

# CREATING A GUROBI MODEL
mod1 = Model()

# CREATE VARIABLES    
vars = {}
for i in range(n):
   for j in range(i+1):
     vars[i,j] = mod1.addVar(obj=dj_dist_matrix[i,j], vtype=GRB.BINARY,
                          name='e'+str(i)+'_'+str(j))
     vars[j,i] = vars[i,j]
   mod1.update()
   
# ADDING DEGREE CONSTRAINTS
for i in range(n):
  mod1.addConstr(quicksum(vars[i,j] for j in range(n)) == 2)
  vars[i,i].ub = 0

# OPTIMIZING THE MODEL
mod1.optimize()

# GETTING SELECTED NODES FROM THE SOLUTION
solution1 = mod1.getAttr('x', vars)
selected1 = [(i,j) for i in range(n) for j in range(n) if solution1[i,j] > 0.5]

# CONVERTING SELECTED NODES TO A LIST
for i in range(len(selected1)):
    selected1[i] = list(selected1[i])

# CREATING EDGES TO PLOT
# creating a pandas dataframe of the distance matrix
dj_df1 = pd.DataFrame(dj_dist_matrix)
# creating a edges dataframe 
edges = dj_df1.stack().reset_index()
edges.columns = ['node1', 'node2','weight']

# PLOTTING THE GRAPH WITH SUBTOURS
# building the graph
dj_G1 = nx.from_pandas_dataframe(edges, 'node1', 'node2')
# adding nodes and edges to the plot 
plt.figure(figsize=(12,12))
nx.draw_networkx_nodes(dj_G1, with_labels = True, node_size = 80, font_size = 8, pos=cities_dict, color='red')
nx.draw_networkx_edges(dj_G1, pos=cities_dict, edgelist = selected1)

# ==================== Part (ii) and (iii) ====================

# FUNCTION TO ELIMINATE THE SUB-TOURS
def subtourelim(model, where):    
    if where == GRB.callback.MIPSOL:
        selected_edges = []
        
        # creating a list of edges selected in the solution
        for i in range(len(dj_dist_matrix)):
            solution_sub = model.cbGetSolution([model._vars[i,j] for j in range(len(dj_dist_matrix))])
            selected_edges += [(i,j) for j in range(len(dj_dist_matrix)) if solution_sub[j] > 0.5]
    
        # finding the shortest cycle in the selected edge list
        tour = subtour(selected_edges)
        if len(tour) < len(dj_dist_matrix):
            # adding sub-tour elimination constraint
            expr = 0
            for i in range(len(tour)):
                for j in range(i+1, len(tour)):
                    expr += model._vars[tour[i], tour[j]]
            model.cbLazy(expr <= len(tour)-1)


# FUNCTION TO FIND THE SHORTEST SUB-TOUR GIVEN A LIST OF TOURS
def subtour(edges):
    visited = [False]*len(dj_dist_matrix)
    cycles = []
    lengths = []
    selected = [[] for i in range(len(dj_dist_matrix))]
    for x,y in edges:
        selected[x].append(y)
    while True:
        current = visited.index(False)
        thiscycle = [current]
        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]
            if len(neighbors) == 0:
                break
            current = neighbors[0]
            thiscycle.append(current)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == len(dj_dist_matrix):
            break
    return cycles[lengths.index(min(lengths))]

# CREATING A GUROBI MODEL
mod2 = Model()

# CREATING VARIABLES
vars = {}
for i in range(len(dj_dist_matrix)):
    for j in range(i+1):
        vars[i,j] = mod2.addVar(obj=dj_dist_matrix[i,j], vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))
        vars[j,i] = vars[i,j]
    mod2.update()

# ADDING DEGREE-2 CONSTRAINTS AND FORBIDDING LOOPS
for i in range(len(dj_dist_matrix)):
    mod2.addConstr(quicksum(vars[i,j] for j in range(len(dj_dist_matrix))) == 2)
    vars[i,i].ub = 0
mod2.update()

# OPTIMIZING THE MODEL
mod2._vars = vars
mod2.params.LazyConstraints = 1
mod2.optimize(subtourelim)

# GETTING SELECTED NODES FROM THE SOLUTION
solution = mod2.getAttr('x', vars) # producing list of solutions
selected_nodes = [(i,j) for i in range(len(dj_dist_matrix)) for j in range(len(dj_dist_matrix)) if solution[i,j] > 0.5] #producing list of selected nodes from the solutions

# ==================================================== 2(d) ==================================================== #

# PLOTTING THE TRAVELLING SALESMAN TOUR ON GRAPH

# converting selected nodes to a list
selected_nodes_graph = selected_nodes
for i in range(len(selected_nodes_graph)):
    selected_nodes_graph[i] = list(selected_nodes_graph[i])

# CREATING EDGES TO PLOT
# creating a pandas dataframe of the distance matrix
dj_df2 = pd.DataFrame(dj_dist_matrix)
# creating a pandas dataframe of the distance matrix
edges_tsp = dj_df2.stack().reset_index()
edges_tsp.columns = ['node1', 'node2','weight']

# PLOTTING THE GRAPH
# building the graph
dj_G_tsp =nx.from_pandas_dataframe(edges_tsp, 'node1', 'node2')
# adding nodes and edges to the graph
plt.figure(figsize=(12,12))
nx.draw_networkx_nodes(dj_G_tsp, with_labels = True, node_size = 100, font_size = 8, pos=cities_dict)
nx.draw_networkx_edges(dj_G_tsp, pos=cities_dict, edgelist = selected_nodes_graph)


# CREATING THE TRAVELLING SALESMAN TOUR ON DJIBOUTI MAP
# setting size of the map
plt.figure(figsize=(12,12))
# using imageservice for a 'better' graph
dj_tsp_map = Basemap(llcrnrlon=41.75, llcrnrlat=10.94, urcrnrlon=43.4, urcrnrlat=12.72, epsg=4713)
dj_tsp_map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= False)
# assigning X & Y co-ordinates for plotting on the map
x, y = dj_tsp_map(dj_tsp_df['longitude'].values, dj_tsp_df['latitude'].values)
# looping thru the edges to be displayed
for n in range (len(selected_nodes)):
    if selected_nodes[n][0] < selected_nodes[n][1]:
        node1 = selected_nodes[n][0]
        node2 = selected_nodes[n][1]
        dj_tsp_map.drawgreatcircle(dj_tsp_df['longitude'][node1], dj_tsp_df['latitude'][node1],
                                   dj_tsp_df['longitude'][node2], dj_tsp_df['latitude'][node2],
                                   linewidth = 2,color = 'r')
# creating a scatter plot
dj_tsp_map.scatter(x, y, s=100, edgecolor='black', marker = 'o', color = 'r')
# labels = [left,right,top,bottom]
dj_tsp_map.drawparallels(parallels,labels=[True,True,True,True])
dj_tsp_map.drawmeridians(meridians,labels=[True,True,True,True])
plt.show()

# PRINTING THE SEQUENCE OF NODES OF THE TSP TOUR
tsp_tour = [0]
k = 0
while k < (len(selected_nodes)/2 - 1):
    for n in range(len(selected_nodes)):
        if (selected_nodes[n][0] == tsp_tour[-1] and selected_nodes[n][1] not in tsp_tour):
            tsp_tour.append(selected_nodes[n][1])
            k += 1  
print(tsp_tour)
