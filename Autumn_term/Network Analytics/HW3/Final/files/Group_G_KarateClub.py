#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:59:29 2017

@author: Veronique
"""


from pandas import Series, DataFrame
import pandas as pd
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import community
import igraph
from networkx.algorithms.community import centrality
import itertools

# create graph
club = r'./zachary_karate_club.txt'
G = nx.read_edgelist(club, nodetype = str)

# rough idea of the network
labels_dict = dict()
for i in list(G.nodes()):
    labels_dict[i] = str(int(i)+1)
nx.draw(G, 
        pos = nx.spring_layout(G, k = 0.25),
        node_size = list(55 * i for i in dict(nx.degree(G, G.nodes())).values()),
        node_color = 'tomato',
        edge_color = 'grey',
        alpha = 0.7,
        with_labels = True,
        labels = labels_dict)
plt.show()

nx.draw(G, 
        pos = nx.shell_layout(G),
        node_size = list(55 * i for i in dict(nx.degree(G, G.nodes())).values()),
        node_color = 'tomato',
        edge_color = 'grey',
        alpha = 0.7,
        with_labels = True,
        labels = labels_dict)
plt.show()

#------------------------------------------------------------------
# 1.(2) - challenge - community
# the Louvain community package that finds best modularity partition via some fast heuristics. 
# Note that the class data and the village data is in a 0-1 matrix format.
# You may want to transform it to adjlist format  write to file and reread, or read each line and add one edge at a time 


# get modularity - Find a partition that maximizes modularity
part = best_partition(G)
mod = modularity(part, G)
print("Louvain - modularity: ", mod)


# dendrogram - partition at certain level
den = generate_dendrogram(G)
for level in range(len(den)):
   print("partition at level", level, " is ", partition_at_level(den, level), "\n")
print("The least partition gives 4 communities, and the second least gives 7 communities.")


# 2. Plot using networkx where different community households have different colors.  
# for 1.(2)
# level 0
lv0_dict = partition_at_level(den, 0)
lv0_part_dict = dict()
for i in list(set(lv0_dict.values())):
    lv0_part_dict[i] = list()
    for j in list(lv0_dict.keys()):
        if lv0_dict[j] == i:
            lv0_part_dict[i].append(j)
values = [lv0_dict[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, 
        pos, 
        cmap = plt.get_cmap('jet'), 
        node_color = values, 
        node_size = 300, 
        alpha = 0.6,
        edge_color = 'lightgray',
        with_labels = True)
plt.title('Karate Club Community - Louvain Partition lv0')

for i in list(lv0_part_dict.keys()):
    print("The " + str(int(i)+1) + "th community contains " + str(len(lv0_part_dict[i])) + " nodes: " + str(lv0_part_dict[i]) + ". \n")

# too many communities

# level 1
lv1_dict = partition_at_level(den, 1)
lv1_part_dict = dict()
for i in list(set(lv1_dict.values())):
    lv1_part_dict[i] = list()
    for j in list(lv1_dict.keys()):
        if lv1_dict[j] == i:
            lv1_part_dict[i].append(j)
values = [lv1_dict[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, 
        pos, 
        cmap = plt.get_cmap('jet'), 
        node_color = values, 
        node_size = 300, 
        alpha = 0.6,
        edge_color = 'lightgray',
        with_labels = True)
plt.title('Karate Club Community - Louvain Partition lv1')


for i in list(lv1_part_dict.keys()):
    print("The " + str(int(i)+1) + "th community contains " + str(len(lv1_part_dict[i])) + " nodes: " + str(lv1_part_dict[i]) + ". \n")


