# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:18:54 2017

@author: 64254
"""
import pandas as pd
import numpy as np
import networkx as nx
import glob # for looping file in an document
import re # for filename extract



village1 = pd.read_csv('adj_allVillageRelationships_HH_vilno_55.csv', header = None)
village1.info()
village1.shape[0]
village1.iloc[1,1]
#compute eigenvector centrality of one village
#1. create an adjacency matrix for the village
village1_adjacency_matrix = np.zeros(shape = village1.shape)
for i in range(village1.shape[0]):
    for j in range(village1.shape[1]):
        village1_adjacency_matrix[i, j] = village1.iloc[i,j]
#2.compute the eigenvector centrality of the village
G=nx.from_numpy_matrix(village1_adjacency_matrix)
a = nx.eigenvector_centrality_numpy(G)
type(a)
#looping all the document to compute eigenvector centrality for each village
list1 = list()
path = "/Users/64254/Documents/IC/Network Analytics/homewrok3/relationship/*.csv"
for fname in glob.glob(path):
    village = pd.read_csv(fname, header = None)
    village_adjacency_matrix = np.zeros(shape = village.shape)
    for i in range(village.shape[0]):
        for j in range(village.shape[1]):
            village_adjacency_matrix[i, j] = village.iloc[i,j]
    G=nx.from_numpy_matrix(village_adjacency_matrix)
    a = nx.eigenvector_centrality_numpy(G)
    res = re.findall("vilno_(\d+).csv", fname)
    dict1 = {}
    dict1[res[0]] = a
    list1.append(dict1)

eigenvector_centrality = list1
type(eigenvector_centrality[0]['1'])
eigenvector_centrality.sort(key = lambda s: int(list(s.keys())[0])) # sort the dictionanry in the list according to villiage
# create a dataframe

df = pd.DataFrame()
for i in range(len(eigenvector_centrality)):
    for k,v in eigenvector_centrality[i].items():
        for j,s in v.items():
            df = df.append({'eigenvalue_centrality': s,
                'village': k,
                'admatrix': int(j+1)
                }, ignore_index=True)
    
# convert to a csv file
df.to_csv("eigenvector_centrality.csv", sep=',', encoding='utf-8')
# adding leader column and HHsurvey column into the file by hand and form "eigenvector_centrality_withleader.csv"
#compute leader eigenvector centrality
ec = pd.read_csv('eigenvector_centrality_withleader.csv', index_col = 0)
sc = pd.read_csv('cross_sectional.csv')
leader_ec = ec.loc[(ec['leader'] == 1)][['village', 'eigenvalue_centrality', 'hhSurveyed','leader']]
bss_vil = leader_ec.loc[leader_ec['village'].isin(sc['village'].tolist())]# use the 45 villages in cross_sectional.xlsx
cross_sectional1 = bss_vil.groupby(['village']).mean()
# the cross_sectional table with leader eigenvector centrality
cross_sectional = cross_sectional1[['eigenvalue_centrality']] 

#2. leader degree

degree = dict(G.degree())
type(degree)
#looping all the document to compute degree for each village
list2 = list()
path = "/Users/64254/Documents/IC/Network Analytics/homewrok3/relationship/*.csv"
for fname in glob.glob(path):
    village = pd.read_csv(fname, header = None)
    village_adjacency_matrix = np.zeros(shape = village.shape)
    for i in range(village.shape[0]):
        for j in range(village.shape[1]):
            village_adjacency_matrix[i, j] = village.iloc[i,j]
    G=nx.from_numpy_matrix(village_adjacency_matrix)
    degree = dict(G.degree())
    res = re.findall("vilno_(\d+).csv", fname)
    dict1 = {}
    dict1[res[0]] = degree
    list2.append(dict1)

degree_all = list2

degree_all.sort(key = lambda s: int(list(s.keys())[0])) # sort the dictionanry in the list according to villiage
# create a dataframe

df1 = pd.DataFrame()
for i in range(len(degree_all)):
    for k,v in degree_all[i].items():
        for j,s in v.items():
            df1 = df1.append({'degree': s,
                'village': k,
                'admatrix': int(j+1)
                }, ignore_index=True)
# add degree to eigenvector_centrality_withleader(this is the file will contain information of all villages)
csv_input = ec
csv_input['degrees'] = df1[['degree']]
csv_input.to_csv('villages.csv', index = 0)
leader_degree = csv_input.loc[(csv_input['leader'] == 1) & (csv_input['hhSurveyed'] == 1)][['village', 'degrees', 'hhSurveyed','leader']]
bss_vil_de = leader_degree.loc[leader_degree['village'].isin(sc['village'].tolist())]
cross_sectional_de = bss_vil_de.groupby(['village']).mean()


# add--the cross_sectional table with leader degrees
cross_sectional['leader_degrees'] = cross_sectional_de[['degrees']] 

#3.numbers of household
cross_sectional_hh = ec.groupby(['village']).size()# calculate the row of each village
csh = cross_sectional_hh.to_frame() #transfrom the series type to datafrmae
csh.columns = ['household'] # add column name
# add-- the household into the cross sectional table

cross_sectional['household'] = csh['household']


#4. fraction of taking leaders
list3 = []
path = "/Users/64254/Documents/IC/Network Analytics/homewrok3/MF Dummy/*.csv"
for filename in glob.glob(path):
    mf = pd.read_csv(filename, header = None)
    res = re.findall("MF(\d+).csv", filename)
    for i in range(mf.shape[0]):
        dict1 = {}
        dict1[res[0]] = mf.to_dict()
    list3.append(dict1)
len(list3)
mf_dummy = list3
mf_dummy.sort(key = lambda s: int(list(s.keys())[0])) # sort the dictionanry in the list according to villiage
#convert it into a dataframe
df4 = pd.DataFrame()# contains 49 mf_dummy data
for i in range(len(mf_dummy)):
    for k,v in mf_dummy[i].items():
        for j,s in v.items():
            for l,m in s.items():
                df4 = df4.append({'take_up': m,
                'village': int(k),
                'admatrix': int(l+1)
                }, ignore_index=True)
#exclude village not in cross_sectonal:
take_up_45 = df4.loc[df4['village'].isin(sc['village'].tolist())]

#exclude village in the village file that not in cross_section file , make it the 45 villages.

csv_input_45 = csv_input.loc[csv_input['village'].isin(sc['village'].tolist())]

csv_input_45.to_csv('villages_45.csv', index = 0)
take_up_45.to_csv('take_up_45.csv', index = 0)
# merge them and form the csv "household_45.csv" by hand in csv directly, then:
hh_45 = pd.read_csv('household_45.csv')
# calculate fraction of taking leaders
hh_45_leader = hh_45.loc[(hh_45['leader']==1)][['village', 'take_up','hhSurveyed','leader']]
take_up_45_leader = hh_45_leader.groupby(['village']).mean()

#add-- fractoin of taking leaders into cross sectional table
cross_sectional['fractoin_of_taking_leaders'] = take_up_45_leader['take_up']

#5.Eigenvector centrality of taking leaders
takeup_leader_45 = hh_45.loc[(hh_45['leader']==1) & (hh_45['take_up']==1)][['village','eigenvalue_centrality','leader','take_up']]
takeup_leader_45_ec = takeup_leader_45.groupby(['village']).mean()
#add--Eigenvector centrality of taking leaders in cross sectional table
cross_sectional['Eigenvector_centrality_taking_leader'] = takeup_leader_45_ec['eigenvalue_centrality']


#6.Savings -- no need

#7. Fraction GM --no need

#8. take up rate
# take up rate for 45 villages's non leader
take_up1 = hh_45.loc[(hh_45['leader'] == 0)][['village', 'take_up', 'hhSurveyed','leader']]
takeup_non_leader = take_up1.groupby(['village']).mean()


#calculate take up rate for 45 village for all household
takeup_45_all = hh_45.groupby(['village']).mean()

# add-- the non_leader_takeup_rate into the cross sectional table
cross_sectional['mf_rate_nonleader'] = takeup_non_leader['take_up']


#9.regression
cross_sectional.columns = ['leader_eigenvector_centrality', 'leader_degrees', 'household',
       'mf_rate_nonleader', 'fraction_of_taking_leaders',
       'Eigenvector_centrality_taking_leader']
cross_sectional.to_csv("cross_sectional_45_gw.csv")

#10.write the report









