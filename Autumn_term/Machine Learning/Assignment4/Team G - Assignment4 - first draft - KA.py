# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:08:40 2017

@author: karim
"""

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

#1. Exploring the website
"""
When looking at the webpage of Argentine players,
it is evident the offset figure rises by an increment of 80.
"""

#2 Explanation of code
"""
The attributes establishes our column vectors that we want to classify.

Links adds all the players that are included within each webpage, up to the 300
players specified. This involves changing the offset increment by 80, obtaining
640 player names (based on the py file). Beautiful soup is used to extract this
information, where we run a for loop to store this information in a list (links)

The next lines of code help establish the code needed to pull up the 
attribute figures (i.e. to categorise the data correctly).

The final for loop runs the links to each player, pulling up all the attribute 
figures and organising this, according to the correct characteristic.

This is then put into a dataframe before being converted into a CSV file.

Note: the py. source code provided on the Hub excludes the first top 10 players, 
and pulls up information for more than 300 players. The latter has been changed, 
given that the question specifically specifies the first 300 players.
"""

#3 Adjustment for England players
"""
There would be two changes compared to pulling up the first 300 Argentine players

1) I would change the na=52 to na=14 in line 22, 
so that it pulls up the names of England players:  
page=requests.get('http://sofifa.com/players?na=14&offset='+offset)

2) For line 21, I would increase the increment of 80 within the
    offset function to ensure this exceeds 500:
        
    for offset in ['0','80','160','240', '320', '400', 480']:
""" 
       
#4 Use of k-means clusters

new_data = pd.read_csv('C:/Users/karim/Documents/Imperial/Machine Learning/ProblemSets/Assignment4/ArgentinaPlayers.csv')

new_data_names=new_data.iloc[:300,1]
new_data300 = new_data.iloc[:300,2:]
new_data300.head()

kmean = KMeans(n_clusters=5)
kmeantest = kmean.fit(new_data300)

new_data300['K_Means_Prediction'] = kmeantest.labels_
new_data300['Name']=new_data_names
new_data300.head()

#5 Assigning meaningful labels
"""
We look at our centroids for our 5 clusters to observe any common patterns.
This enables us to make the following deductions below.

"""

centroids = kmeantest.cluster_centers_
output = pd.DataFrame(centroids)
output.columns = ['Crossing','Finishing','Heading accuracy',
 'Short passing','Volleys','Dribbling','Curve',
 'Free kick accuracy','Long passing','Ball control','Acceleration',
 'Sprint speed','Agility','Reactions','Balance',
 'Shot power','Jumping','Stamina','Strength',
 'Long shots','Aggression','Interceptions','Positioning',
 'Vision','Penalties','Composure','Marking',
 'Standing tackle','Sliding tackle','GK diving',
 'GK handling','GK kicking','GK positioning','GK reflexes']

output.iloc[:,0:10]
output.iloc[:,10:20]
output.iloc[:,20:30]
output.iloc[:,30:34]

"""
Likely positions: 
Goalkeeping = 1
Striker = 2
Winger / Midfielder = 0
Defender = 3
Midfielder = 4

We confirm this by examining the data for each footballer within
our dataframe:
"""

new_data300['Category'] = new_data300.K_Means_Prediction.apply(
        lambda x: "Defender / Midfielder" if x==0 
        else "Goalkeeper" if x==1
        else 'Striker' if x==2
        else 'Defender' if x==3
        else 'Midfielder / Attacker')

new_data300.head()

#6 Predicting a position using Centroids

centroids = kmeantest.cluster_centers_
output = pd.DataFrame(centroids)
output.columns = ['Crossing','Finishing','Heading accuracy',
 'Short passing','Volleys','Dribbling','Curve',
 'Free kick accuracy','Long passing','Ball control','Acceleration',
 'Sprint speed','Agility','Reactions','Balance',
 'Shot power','Jumping','Stamina','Strength',
 'Long shots','Aggression','Interceptions','Positioning',
 'Vision','Penalties','Composure','Marking',
 'Standing tackle','Sliding tackle','GK diving',
 'GK handling','GK kicking','GK positioning','GK reflexes']

output.iloc[:,0:10]
output.iloc[:,10:20]
output.iloc[:,20:30]
output.iloc[:,30:34]

revisedoutput = output[['Crossing', 'Sprint speed', 
                        'Long shots', 'Aggression',
                        'Marking', 'Finishing',
                        'GK handling']]


player1b = [45, 40, 35, 45, 60, 40, 15]

len(revisedoutput)
player_dist=np.zeros((5,7))
player_dist = player_dist+player1b
revisedoutput

for index in range(len(revisedoutput)):
    player_dist=revisedoutput - player_dist

player_dist

"""
Comparing the distance, this appears to be consistently lower for cluster 3 (Defender). Therefore,
the player is classified as such.

"""
