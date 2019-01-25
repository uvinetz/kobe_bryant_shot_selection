# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:41:57 2016

@author: Uriel Vinetz and Nir Basanchik
"""
"""
This script predicts the probability of 5000 among all carrer shots Kobe Bryant
ever tried based ONLY on his previous shots. We first selected an SVM model,
carefully choosing the parameters in R. Then, we processed the data and trained
the model in Python. 
"""

import pandas
import numpy as np
#import seaborn as sns


#Reading the data
data = pandas.read_csv("data.csv")

#Some graphic insights
#sns.countplot(x='shot_type', hue='shot_made_flag', data=data)
#sns.countplot(x='shot_zone_area', hue='shot_made_flag', data=data)
#sns.countplot(x='shot_zone_basic', hue='shot_made_flag', data=data)
#sns.countplot(x='shot_zone_range', hue='shot_made_flag', data=data)
#facet = sns.FacetGrid(data, hue="shot_made_flag",aspect=4)
#facet.map(sns.kdeplot,'shot_distance',shade= True)
#facet.set(xlim=(0, data['shot_distance'].max()))
#facet.add_legend()

#Shot made percentage according to different categories.
#print(pandas.pivot_table(data=data,values='shot_made_flag', index=['shot_zone_area'],columns=['shot_type'] , aggfunc=np.mean))
#print(pandas.pivot_table(data=data,values='shot_made_flag', index=['shot_zone_range'], aggfunc=np.mean))
#print(pandas.pivot_table(data=data,values='shot_made_flag', index=['opponent'], aggfunc=np.mean))
#print(pandas.pivot_table(data=data,values='shot_made_flag', index=['season'], aggfunc=np.mean))

#Creating a variable "Hot hand" if in the same game Kobe made his last 3 shots
#We will define a function that updates this variable sice  we would like to do
#it each time we make a prediction

n = len(data) #Number of observations
data["hot_hand"] = [0]*n #Creates the "Hot hand" array with value 0
def update_hot():
    count=0 #Variable for counting scoring streak
    current_game=0
    for index, obs in data.iterrows(): #Iterates every row in the data frame      
        if index>2 and count>=3 and obs["game_id"]==current_game: #If in the current game Kobe is in a streak of 3 scores in a row
            data.set_value(index,"hot_hand",1) #Kobe has hot hand
        if obs["shot_made_flag"]>0: #Kobe scored!
            if obs["game_id"]==current_game: #We didn't moved to next game
                count=count+1 #Add one to the streak
            else:
                count=1 #Scored the first shot in the game
        else:
            count=0 #Reset the streak to 0
        current_game=obs["game_id"] #Update current game
update_hot()

#Now, let's check if Kobe shoots better with the hot hand
#print(pandas.pivot_table(data=data,values='shot_made_flag', index=['hot_hand'], aggfunc=np.mean))
#Surprisingly, Kobe has an overall smaller percentage after a scoring streak.

#Now, we will create a "venue" variable that indicates if the game is home or away
data["venue"] = ["Home"]*n
for index, obs in data.iterrows():
    if obs["matchup"].find('@') != -1: #If there is not an @ it is Home
        data.set_value(index,"venue","Away")

#We will also create a variable "recent_avg" to know if Kobe is in a great moment
#First we will change the date format into sth more useful
from datetime import datetime
for index, obs in data.iterrows():
    x = datetime.strptime(obs["game_date"], "%Y-%m-%d")
    data.set_value(index,"game_date",x)
    
#We will use a 10 day frame to recent average
import datetime
data["recent_avg"]=[0.0]*n
for index, obs in data.iterrows():
    ten_days = pandas.date_range(end=obs["game_date"]-datetime.timedelta(days = 1),periods=10,freq='D').tolist() #Creates an array with the dates of previous 10 days
    subset = data[data["game_date"].isin(ten_days)] #Takes the shot during the last 10 days
    x = np.mean(subset["shot_made_flag"]) #Calculates the scoring average   
    data.set_value(index,"recent_avg",x) #Updates de value in the data
data.recent_avg[np.isnan(data.recent_avg)] = 0.0
#Editing the data to use in prediction algorithms
data.period = data.period.astype('category') #Converts the period to be treated as not numerical    

#We will transform the season array into a sequence of integers and sort the data by season
data.loc[data.season == '1996-97', 'season'] = 1
data.loc[data.season == '1997-98', 'season'] = 2
data.loc[data.season == '1998-99', 'season'] = 3
data.loc[data.season == '1999-00', 'season'] = 4
data.loc[data.season == '2000-01', 'season'] = 5
data.loc[data.season == '2001-02', 'season'] = 6
data.loc[data.season == '2002-03', 'season'] = 7
data.loc[data.season == '2003-04', 'season'] = 8
data.loc[data.season == '2004-05', 'season'] = 9
data.loc[data.season == '2005-06', 'season'] = 10
data.loc[data.season == '2006-07', 'season'] = 11
data.loc[data.season == '2007-08', 'season'] = 12
data.loc[data.season == '2008-09', 'season'] = 13
data.loc[data.season == '2009-10', 'season'] = 14
data.loc[data.season == '2010-11', 'season'] = 15
data.loc[data.season == '2011-12', 'season'] = 16
data.loc[data.season == '2012-13', 'season'] = 17
data.loc[data.season == '2013-14', 'season'] = 18
data.loc[data.season == '2014-15', 'season'] = 19
data.loc[data.season == '2015-16', 'season'] = 20
data = data.sort_values(by=['season','shot_id'])
data.index = range(0,len(data))

#Process numerical variables
num_cols = ['loc_x','loc_y','minutes_remaining','playoffs','seconds_remaining','shot_distance','recent_avg','hot_hand'] #Numerical features
x_num = data[num_cols].as_matrix()
x_num = x_num / np.nanmax(x_num,0) #Feature scaling

#Now categorical features
cat_cols = ['period','action_type','shot_type','opponent','venue']
x_cat = data[cat_cols]
x_cat = x_cat.T.to_dict().values()

#Vectorizing categorical features
from sklearn.feature_extraction import DictVectorizer as DV
vectorizer = DV(sparse = False)
x_cat = vectorizer.fit_transform(x_cat)

#Concatenates numerical and categorical features in a single feature array
new_x = np.hstack((x_num, x_cat))

#Approximates an RBF Kernel before training the model
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
new_x = rbf_feature.fit_transform(new_x)


#Selecting the prediction variable
y = data.shot_made_flag

#Selecting which rows to use for training and testing respectively
test_ind = np.where(np.isnan(y))
test_ind = test_ind[0]
train_ind = np.where(~np.isnan(y))
train_ind = train_ind[0]

from sklearn.linear_model import LogisticRegression
rs = 2505
alg = LogisticRegression(solver='sag',max_iter=10000,random_state=rs,n_jobs=2,verbose=0,tol=0.0001,warm_start=True)

#Now it is time to predict the scoring probablity of each missing shot.
probs = [] #Initialize probability array for missing shots
for i in test_ind:
    if len(probs)-1>=np.where(test_ind==i)[0][0]: continue #If we have already predicted for this observation, skip.
    else:
        #fiveSeasons = train_ind[train_ind[data.season[train_ind]>=data.season[i]-5]==1] #Train only with data of current and two previous seasons
        x_train = new_x[train_ind[train_ind<i]]  #Makes sure to use only previous data            
        y_train = y[train_ind[train_ind<i]]        
        alg.fit(x_train,y_train) #Fit the algorithm to the training data        
        k=i
        while (np.isnan(y[k]) and len(probs)<5000):           
            probs.append(alg.predict_proba(new_x[k].reshape(1,-1))[0][1]) #Predict the probability of scoring the current shot
            k = k+1
            print("Adding prediction number: ", len(probs))

#Preparing the submission file
test_ind = [data.shot_id[j] for j in test_ind]  #Changing the indices to fit the real ones. 
submission = np.vstack((test_ind, probs)).T #Transposing the matrix
submission = pandas.DataFrame(submission) #Converting the matrix to a Data Frame
submission.columns=['shot_id','shot_made_flag'] #Adding the column titles
submission.shot_id = submission.shot_id.astype(int) #Converting the shot id to integer
submission = submission.sort_values(by=['shot_id'])
submission.to_csv(path_or_buf='submission.csv',index=False) #Exporting to a CSV file