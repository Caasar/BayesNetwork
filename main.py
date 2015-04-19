# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:00:41 2015

@author: Caasar
"""
# -*- coding: utf-8 -*-
"""
Use a Bayes Inference algorithm to build a model supporting a simple solution
for the missing data problem.

I assume conditional independence of the provided columns given the clicks.
This allows for a fast inference with low memory requirements.

@author: Fabian Schmieder
"""

import numpy as np
from numpy.random import randn
from network import BayesNetwork, BayesAdaptStemmingNode, BayesCategoricalNode

path = 'data_recruiting_bi_data_challenge.csv'
TEST_SIZE = 50000
cols = np.arange(1,11)
dtypes = [("hotel_id",np.uint64),("city_id",np.uint64),("clicks",float),
         ("stars",float),("distance_to_center",float),("avg_price_hotel",float),
         ("rating",float),("nmbr_partners_index",float),
         ("avg_rel_saving",float),("avg_rank",float)]

data = np.genfromtxt(path,delimiter=',',dtype=dtypes,usecols=cols,names=True)

missing = np.isnan(data['clicks'])

unkown = data[missing]
known  = data[~missing]

# split for own validation
rndset = randn(len(known)).argsort()
test = known[rndset[:TEST_SIZE]]
train = known[rndset[TEST_SIZE:]]

def measure(true,pred):
    """
    Computes the provided weighted MSE performance measure.
    
    Parameters
    ----------
    true : numpy.ndarray
        The true clicks.
    pred : numpy.ndarray
        The predicted clicks.
    """
    w = np.log(true+1)+1
    w /= w.sum()
    diff = true-pred
    return np.inner(w,diff*diff)

netcost = lambda test, net : np.sqrt(measure(test['clicks'],net.mean_prediction(test)))

# %% Set network parameters and train models

nodes = [(BayesAdaptStemmingNode,{'label':'avg_rank','stems':64,'min_raise':5}),
         (BayesAdaptStemmingNode,{'label':'avg_rel_saving','stems':32}),
         (BayesAdaptStemmingNode,{'label':'nmbr_partners_index','stems':32}),
         (BayesAdaptStemmingNode,{'label':'avg_price_hotel','stems':32}), 
         (BayesAdaptStemmingNode,{'label':'distance_to_center','stems':16}), 
         (BayesAdaptStemmingNode,{'label':'rating','stems':32,'min_raise':1}),
         (BayesCategoricalNode,{'label':'city_id'}),
         (BayesCategoricalNode,{'label':'stars','ids':[0,1,2,3,4,5]}),
         (BayesAdaptStemmingNode,{'label':'clicks','stems':64,'min_raise':5})]

parents = {'rating':['clicks'], 'avg_rank':['clicks'],'stars':['clicks'],
           'avg_rel_saving':['clicks'],'nmbr_partners_index':['clicks'],
           'avg_price_hotel':['clicks'],'distance_to_center':['clicks'],
           'distance_to_center':['clicks'],'city_id':['clicks']}


ghost_sample = 1.0
weighting = lambda d: np.log(d['clicks']+1)+1
network = BayesNetwork(nodes,parents).learn(train,ghost_sample,weighting)

w = np.log(test['clicks']+1)+1
pred_baseline = np.inner(w,test['clicks'])/w.sum()
print('Baseline rmse: %.3f' % np.sqrt(measure(test['clicks'],pred_baseline)))
print('All models: %.3f' % netcost(test,network))
