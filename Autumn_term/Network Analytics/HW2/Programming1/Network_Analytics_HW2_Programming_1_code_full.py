#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:28:47 2017

@author: Veronique
"""

from pandas import Series, DataFrame
import pandas as pd
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import reduce
import collections
import seaborn as sns 

# Part 1: Data Cleaning & Drawing

sent_mat = pd.read_csv('/Users/Veronique/Desktop/ICBA/Nov_Dec Term/Network Analytics/Homework/Hw2/HW2_who_talks_to_whom_sent.csv')
received_mat = pd.read_csv('/Users/Veronique/Desktop/ICBA/Nov_Dec Term/Network Analytics/Homework/Hw2/HW2_who_talks_to_whom_received.csv')
avg_mat = pd.read_csv('/Users/Veronique/Desktop/ICBA/Nov_Dec Term/Network Analytics/Homework/Hw2/HW2_who_talks_to_whom_avg.csv')

# re-index the matrix
# col_naming_dict = {str(i+1):i for i in list(range(81))}
# sent_mat.rename(columns = col_naming_dict)
# received_mat

# create node list
node_no_dict = {i:str(i+1) for i in list(range(81))}

sent_node_list = list()
for i in list(range(81)):
    for j in list(range(81)):
        if sent_mat.iloc[i, j] != 0:
            sent_node_list.append([node_no_dict[i], node_no_dict[j], sent_mat.iloc[i, j]])

received_node_list = list()
for i in list(range(81)):
    for j in list(range(81)):
        if received_mat.iloc[i, j] != 0:
            received_node_list.append([node_no_dict[i], node_no_dict[j], received_mat.iloc[i, j]])

avg_node_list = list()
for i in list(range(81)):
    for j in list(range(81)):
        if avg_mat.iloc[i, j] != 0:
            avg_node_list.append([node_no_dict[i], node_no_dict[j], avg_mat.iloc[i, j]])

# create DiGraph
G_sent = nx.DiGraph()
G_sent.add_weighted_edges_from(sent_node_list)

G_received = nx.DiGraph()
G_received.add_weighted_edges_from(received_node_list)

G_total = nx.DiGraph()
G_total.add_weighted_edges_from(avg_node_list)

# draw the graphs
# sent (weight > k)
k_sent = 0

node_color_vec_sent = dict()
for key in G_sent.nodes():
    if len(G_sent.neighbors(key)) >= np.percentile([len(G_sent.neighbors(i)) for i in G_sent.nodes()], 90):
        node_color_vec_sent[key] = 'r'
    elif len(G_sent.neighbors(key)) >= np.percentile([len(G_sent.neighbors(i)) for i in G_sent.nodes()], 75):
        node_color_vec_sent[key] = 'hotpink'
    else:
        node_color_vec_sent[key] = 'pink'
        
data_sent = {'edgelist': [sent_node_list[i] for i in list(range(2046))],
            'tail': [sent_node_list[i][0] for i in list(range(2046))],
            'head': [sent_node_list[i][1] for i in list(range(2046))],
            'weight': [sent_node_list[i][2] for i in list(range(2046))],
            'num_neighbor': [len(G_sent.neighbors(i[0])) for i in sent_node_list],
            'color': [node_color_vec_sent[i[0]] for i in sent_node_list]}
node_attr_sent = DataFrame(data_sent)

node_attr_sent_draw = DataFrame(columns = ['edgelist', 'tail', 'head', 'weight', 'num_neighbor', 'color'])
for i in list(range(2046)):
    if node_attr_sent['weight'][i] > k_sent:
        node_attr_sent_draw = node_attr_sent_draw.append(node_attr_sent.loc[i], ignore_index=True)

nx.draw(G_sent, 
        pos = nx.spring_layout(G_sent, k = 0.5, iterations = 80, scale = 900),
        edgelist = list(node_attr_sent_draw['edgelist']),
        with_labels = True,
        node_size = [(len(G_sent.neighbors(i)) - 2) * 35 for i in G_sent.nodes()],
        node_color = list(node_color_vec_sent.values()),
        edge_color = 'darkgrey',
        width = [float(d['weight'] / 24 ) for (u, v, d) in G_sent.edges(data = True)],
        alpha = 0.4,
        font_size = 9,
        arrows = False)
plt.title('Main Information Sender Network',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Main Information Sender Network',
            bbox_inches='tight')


# received (weight > k)
k_received = 0

node_color_vec_received = dict()
for key in G_received.nodes():
    if len(G_received.neighbors(key)) >= np.percentile([len(G_received.neighbors(i)) for i in G_received.nodes()], 90):
        node_color_vec_received[key] = 'r'
    elif len(G_received.neighbors(key)) >= np.percentile([len(G_received.neighbors(i)) for i in G_received.nodes()], 75):
        node_color_vec_received[key] = 'hotpink'
    else:
        node_color_vec_received[key] = 'pink'
        
data_received = {'edgelist': [received_node_list[i] for i in list(range(1985))],
            'tail': [received_node_list[i][0] for i in list(range(1985))],
            'head': [received_node_list[i][1] for i in list(range(1985))],
            'weight': [received_node_list[i][2] for i in list(range(1985))],
            'num_neighbor': [len(G_received.neighbors(i[0])) for i in received_node_list],
            'color': [node_color_vec_received[i[0]] for i in received_node_list]}
node_attr_received = DataFrame(data_received)

node_attr_received_draw = DataFrame(columns = ['edgelist', 'tail', 'head', 'weight', 'num_neighbor', 'color'])
for i in list(range(1985)):
    if node_attr_received['weight'][i] > k_received:
        node_attr_received_draw = node_attr_received_draw.append(node_attr_received.loc[i], ignore_index=True)

nx.draw(G_received, 
        pos = nx.spring_layout(G_received, k = 0.6, iterations = 80, scale = 900),
        edgelist = list(node_attr_received_draw['edgelist']),
        with_labels = True,
        node_size = [(len(G_received.neighbors(i)) - 2) * 35 for i in G_received.nodes()],
        node_color = list(node_color_vec_received.values()),
        edge_color = 'darkgrey',
        width = [float(d['weight'] / 24 ) for (u, v, d) in G_received.edges(data = True)],
        alpha = 0.4,
        font_size = 9,
        arrows = False)
plt.title('Main Information Receiver Network',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Main Information Receiver Network',
            bbox_inches='tight')



# total (weight > k)
k_total = 0

node_color_vec_total = dict()
for key in G_total.nodes():
    if len(G_total.neighbors(key)) >= np.percentile([len(G_total.neighbors(i)) for i in G_total.nodes()], 90):
        node_color_vec_total[key] = 'r'
    elif len(G_total.neighbors(key)) >= np.percentile([len(G_total.neighbors(i)) for i in G_total.nodes()], 75):
        node_color_vec_total[key] = 'hotpink'
    else:
        node_color_vec_total[key] = 'pink'

data_total = {'edgelist': [avg_node_list[i] for i in list(range(2057))],
              'tail': [avg_node_list[i][0] for i in list(range(2057))],
              'head': [avg_node_list[i][1] for i in list(range(2057))],
              'weight': [avg_node_list[i][2] for i in list(range(2057))],
              'num_neighbor': [len(G_total.neighbors(i[0])) for i in avg_node_list],
              'color': [node_color_vec_total[i[0]] for i in avg_node_list]}
node_attr_total = DataFrame(data_total)

node_attr_total_draw = DataFrame(columns = ['edgelist', 'tail', 'head', 'weight', 'num_neighbor', 'color'])
for i in list(range(2057)):
    if node_attr_total['weight'][i] > k_total:
        node_attr_total_draw = node_attr_total_draw.append(node_attr_total.loc[i], ignore_index=True)

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.5, iterations = 85, scale = 900),
        # pos = nx.random_layout(G_total),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [(len(G_total.neighbors(i)) - 2) * 35 for i in G_total.nodes()],
        node_color = list(node_color_vec_total.values()),
        edge_color = 'darkgrey',
        width = [float(d['weight'] / 18 ) for (u, v, d) in G_total.edges(data = True)],
        alpha = 0.4,
        font_size = 9,
        arrows = False)
plt.title('Total Conversation Network',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Total Conversation Network',
            bbox_inches='tight')

# Part 2: Network Analysis
# essentially calculating centrality measures (try at least one eigenvalue based one)
# degree centrality measure
degree_ctr_total = nx.degree_centrality(G_total)
in_degree_ctr_total = nx.in_degree_centrality(G_total)
out_degree_ctr_total = nx.out_degree_centrality(G_total)

degree_ctr_sent = nx.degree_centrality(G_sent)
degree_ctr_received = nx.degree_centrality(G_received)

DC_Counter = collections.Counter(degree_ctr_total)
plt.hist([value for key, value in DC_Counter.items()], 
          bins = 40, 
          color = 'darkgreen', 
          histtype = "stepfilled")
plt.title('Histogram of Degree Centrality Distribution',
          fontweight="bold",
          fontsize = 16)
plt.savefig('Histogram of Degree Centrality Distribution')

node_color_vec_DC = dict()
for key in G_total.nodes():
    if degree_ctr_total[key] >= np.percentile(list(degree_ctr_total.values()), 95):
        node_color_vec_DC[key] = 'midnightblue'
    elif degree_ctr_total[key] >= np.percentile(list(degree_ctr_total.values()), 75):
        node_color_vec_DC[key] = 'dodgerblue'
    else:
        node_color_vec_DC[key] = 'lightskyblue'

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.5, iterations = 55, scale = 3000),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [3000 * i ** 2 - 300 for i in list(degree_ctr_total.values())],
        node_color = list(node_color_vec_DC.values()),
        edge_color = 'darkgrey',
        alpha = 0.6,
        font_size = 9,
        arrows = False)
plt.title('Size and Color by Degree Centrality',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Size and Color by Degree Centrality',
            bbox_inches='tight')


# betweenness centrality measure
betweenness_nodes_ctr_total = nx.betweenness_centrality(G_total) # for nodes
betweenness_edges_ctr_total = nx.edge_betweenness_centrality(G_total) # for edges

BC_Counter = collections.Counter(betweenness_nodes_ctr_total)
plt.hist([value for key, value in BC_Counter.items()], 
          bins = 40, 
          color = 'darkgreen', 
          histtype = "stepfilled")
plt.title('Histogram of Betweenness Centrality Distribution',
          fontweight="bold",
          fontsize = 16)
plt.savefig('Histogram of Betweenness Centrality Distribution')

node_color_vec_BC = dict()
for key in G_total.nodes():
    if betweenness_nodes_ctr_total[key] >= np.percentile(list(betweenness_nodes_ctr_total.values()), 95):
        node_color_vec_BC[key] = 'midnightblue'
    elif betweenness_nodes_ctr_total[key] >= np.percentile(list(betweenness_nodes_ctr_total.values()), 75):
        node_color_vec_BC[key] = 'dodgerblue'
    else:
        node_color_vec_BC[key] = 'lightskyblue'

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.5, iterations = 55, scale = 3000),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [3000 * i ** 2 - 300 for i in list(degree_ctr_total.values())],
        node_color = list(node_color_vec_BC.values()),
        edge_color = 'darkgrey',
        alpha = 0.6,
        font_size = 9,
        arrows = False)
plt.title('Size and Color by Betweenness Centrality',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Size and Color by Betweenness Centrality',
            bbox_inches='tight')


# closeness centrality measure
closeness_ctr_total = nx.closeness_centrality(G_total)

CC_Counter = collections.Counter(closeness_ctr_total)
plt.hist([value for key, value in CC_Counter.items()], 
          bins = 40, 
          color = 'darkgreen', 
          histtype = "stepfilled")
plt.title('Histogram of Closeness Centrality Distribution',
          fontweight="bold",
          fontsize = 16)
plt.savefig('Histogram of Closeness Centrality Distribution')

node_color_vec_CC = dict()
for key in G_total.nodes():
    if closeness_ctr_total[key] >= np.percentile(list(closeness_ctr_total.values()), 95):
        node_color_vec_CC[key] = 'midnightblue'
    elif closeness_ctr_total[key] >= np.percentile(list(closeness_ctr_total.values()), 75):
        node_color_vec_CC[key] = 'dodgerblue'
    else:
        node_color_vec_CC[key] = 'lightskyblue'

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.5, iterations = 55, scale = 3000),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [15000 * i - 7500 for i in list(closeness_ctr_total.values())],
        node_color = list(node_color_vec_CC.values()),
        edge_color = 'darkgrey',
        alpha = 0.6,
        font_size = 9,
        arrows = False)
plt.title('Size and Color by Closeness Centrality',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Size and Color by Closeness Centrality',
            bbox_inches='tight')



# Local metrics comparison
plt.hist([(value - min([value for key, value in DC_Counter.items()])) / (max([value for key, value in DC_Counter.items()]) - min([value for key, value in DC_Counter.items()])) for key, value in DC_Counter.items()], 
           bins = 40, 
           color = 'olivedrab',
           alpha = 0.4,
           histtype = "stepfilled",
           label = 'Degree')
plt.hist([(value - min([value for key, value in BC_Counter.items()])) / (max([value for key, value in BC_Counter.items()]) - min([value for key, value in BC_Counter.items()])) for key, value in BC_Counter.items()], 
           bins = 40, 
           color = 'seagreen',
           alpha = 0.4,
           histtype = "stepfilled", 
           label = 'Betweenness')
plt.hist([(value - min([value for key, value in CC_Counter.items()])) / (max([value for key, value in CC_Counter.items()]) - min([value for key, value in CC_Counter.items()])) for key, value in CC_Counter.items()], 
           bins = 40, 
           color = 'darkcyan', 
           alpha = 0.4,
           histtype = "stepfilled",
           label = 'Closeness')
plt.title('(Normalised) Local Metrics Distribution',
          fontweight="bold",
          fontsize = 16)
plt.legend(loc = 'upper left')
plt.savefig('(Normalised) Local Metrics Distribution')

# eigenvector centrality measure
eigen_ctr_total = nx.eigenvector_centrality(G_total)

EC_Counter = collections.Counter(eigen_ctr_total)
plt.hist([value for key, value in EC_Counter.items()], 
          bins = 40, 
          color = 'darkgreen', 
          histtype = "stepfilled")
plt.title('Histogram of EigenCentrality Distribution',
          fontweight="bold",
          fontsize = 16)
plt.savefig('Histogram of EigenCentrality Distribution')

node_color_vec_EC = dict()
for key in G_total.nodes():
    if eigen_ctr_total[key] >= np.percentile(list(eigen_ctr_total.values()), 95):
        node_color_vec_EC[key] = 'midnightblue'
    elif eigen_ctr_total[key] >= np.percentile(list(eigen_ctr_total.values()), 75):
        node_color_vec_EC[key] = 'dodgerblue'
    else:
        node_color_vec_EC[key] = 'lightskyblue'

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.5, iterations = 55, scale = 3000),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [11000 * i for i in list(eigen_ctr_total.values())],
        node_color = list(node_color_vec_EC.values()),
        edge_color = 'darkgrey',
        alpha = 0.6,
        font_size = 9,
        arrows = False)
plt.title('Size and Color by EigenCentrality',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Size and Color by EigenCentrality',
            bbox_inches='tight')

# pangerank
pagerank_total = nx.pagerank(G_total)

pgrk_Counter = collections.Counter(pagerank_total)
plt.hist([value for key, value in pgrk_Counter.items()], 
          bins = 40, 
          color = 'darkgreen', 
          histtype="stepfilled")
plt.title('Histogram of PageRank Distribution',
          fontweight="bold",
          fontsize = 16)
plt.savefig('Histogram of PageRank Distribution')

node_color_vec_pgrk = dict()
for key in G_total.nodes():
    if pagerank_total[key] >= np.percentile(list(pagerank_total.values()), 95):
        node_color_vec_pgrk[key] = 'midnightblue'
    elif pagerank_total[key] >= np.percentile(list(pagerank_total.values()), 75):
        node_color_vec_pgrk[key] = 'dodgerblue'
    else:
        node_color_vec_pgrk[key] = 'lightskyblue'

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.5, iterations = 55, scale = 3000),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [90000 * i for i in list(pagerank_total.values())],
        node_color = list(node_color_vec_pgrk.values()),
        edge_color = 'darkgrey',
        alpha = 0.6,
        font_size = 9,
        arrows = False)
plt.title('Size and Color by PageRank',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Size and Color by PageRank',
            bbox_inches='tight')

# Global metrics comparison
plt.hist([(value - min([value for key, value in EC_Counter.items()])) / (max([value for key, value in EC_Counter.items()]) - min([value for key, value in EC_Counter.items()])) for key, value in EC_Counter.items()], 
           bins = 40, 
           color = 'seagreen',
           alpha = 0.5,
           histtype = "stepfilled",
           label = 'Eigenvector')
plt.hist([(value - min([value for key, value in pgrk_Counter.items()])) / (max([value for key, value in pgrk_Counter.items()]) - min([value for key, value in pgrk_Counter.items()])) for key, value in pgrk_Counter.items()], 
           bins = 40, 
           color = 'darkcyan', 
           alpha = 0.5,
           histtype = "stepfilled",
           label = 'PageRank')
plt.title('(Normalised) Global Metrics Distribution', 
          fontweight = "bold",
          fontsize = 16)
plt.legend(loc = 'upper right')
plt.savefig('(Normalised) Global Metrics Distribution')


# and clustering coefficients and gaining some insight into the network.  
clustering_coef = nx.clustering(G_total.to_undirected())
avg_clustering_coef = nx.average_clustering(G_total.to_undirected())

clstr_Counter = collections.Counter(clustering_coef)
plt.hist([value for key, value in clstr_Counter.items()], 
          bins = 40, 
          color = 'darkgreen', 
          histtype="stepfilled")
plt.title('Histogram of Node Clustering Coefficient',
          fontweight="bold",
          fontsize = 16)
plt.savefig('Histogram of Node Clustering Coefficient')


# find cliques
cliques_total = list(nx.find_cliques(G_total.to_undirected()))

max_cliques = [i for i in cliques_total if len(i) == max([len(i) for i in cliques_total])]
# active ppl in main cliques
max_cliques_set = [set(i) for i in max_cliques]
ppl_in_all_max_clique = list(reduce(lambda x, y: x.intersection(y), max_cliques_set))

max_clique_color_vec = list(node_color_vec_total.values())
for i in list(range(len(max_clique_color_vec))):
    if str(i + 1) in ppl_in_all_max_clique:
        max_clique_color_vec[i] = 'tomato'
    else:
        max_clique_color_vec[i] = 'peachpuff'

nx.draw(G_total, 
        pos = nx.spring_layout(G_total, k = 0.6, iterations = 85, scale = 900),
        edgelist = list(node_attr_total_draw['edgelist']),
        with_labels = True,
        node_size = [(len(G_total.neighbors(i)) - 2) * 40 for i in G_total.nodes()],
        node_color = max_clique_color_vec,
        edge_color = 'darkgrey',
        width = [float(d['weight'] / 18 ) for (u, v, d) in G_sent.edges(data = True)],
        alpha = 0.5,
        font_size = 9,
        arrows = False)
plt.title('Max Clique in the Network',
          fontweight = "bold",
          fontsize = 18)
plt.savefig('Max Clique in the Network',
            bbox_inches='tight')

# The objective for me (your client) is to identify who are the leaders and opinion-makers
# higher num_neighbor: talk to more people - width
# higher (avg) weight: talk more/deeper to specific people - depth
# higher clustering coef: 
# higher num of triangles (local cluster coef): 
# higher transitivity (global cluster coef):

    