# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:28:00 2015


Uses a Bayes Inference algorithm to build a network supporting a simple 
solution for the missing data problem. In the developed framework the missing 
values will represented with NaNs.

Conditional independence of the provided columns given the clicks is assumed.
This allows for a fast inference with low memory requirements.The developed
BayesNetwork can also be used for non naive solution. However after a
few experiments the naive approach did show the best performance.

The biggest improvement on the performance is due to the use of an adaptive
mapping of the click counts. It can be observed, that  while they cover a large
rangeare they are also highly concentrated at small click counts. This makes
it difficult to map them to discrete values using a linearly spaces bins.
Because of this the `AdaptStemmingNode` was developed to compute a
general data dependent mapping.

Furthermore, the Bayes Network can be trained with all provided samples
including the samples where the clicks are missing. However, this forces
the Bayes Network to use a EM algorithm to for learning. The resulting
network typicly shows a better performance, at the cost of a much larger
training time. The observed improvement for the naive network did not
justify this. Therefore, only the samples with known click counts are used
for training the network.

The network parameters where optimized by a simple cycling grid search
changing each node individually, while retaining the best result on a hold
out test set.

Examples
--------

Run the script from shell

>>> python main.py [infile.csv] [outfile.csv]

@author: Fabian Schmieder
"""

import numpy as np
import sys
from numpy.random import rand
from network import BayesNetwork, AdaptStemmingNode, CategoricalNode, \
                    LinearStemmingNode


def filter_argv(script,inpath='data_recruiting_bi_data_challenge.csv',
                outpath='prediction.csv',*argv):
    return inpath, outpath


def performace_measure(data,pred):
    """
    Computes the provided weighted rMSE performance measure.
    
    Parameters
    ----------
    data : structured numpy.ndarray
        The data used to estimate the performace.
    pred : numpy.ndarray
        The predicted clicks.
    weighting : function
        The weighting to use
    """
    true = data['clicks']
    weights = weighting(data)
    diff = true-pred.astype(int)
    return np.sqrt(np.inner(weights,diff*diff)/weights.sum())
    
def baseline(data):
    """
    Returns the baseline prediction, a simple weighted mean of the avaliabe
    clicks
    """
    weights = weighting(data)
    return np.inner(weights,data['clicks'])/weights.sum()

def weighting(data):
    """
    Returns the weights for the data samples.
    """
    weights = np.log(data['clicks']+1)+1
    unknown = np.isnan(weights)
    if np.any(unknown):
        weights[unknown] = baseline(data[~unknown])
    return weights
    

# %% Set network parameters 

# All features are child nodes of the root 'clicks' node
nodes = [(AdaptStemmingNode,{'label':'avg_rank','stems':32,'parents':['clicks']}),
         (LinearStemmingNode,{'label':'avg_rel_saving','stems':16,'parents':['clicks']}),
         (AdaptStemmingNode,{'label':'nmbr_partners_index','stems':16,'parents':['clicks']}),
         (AdaptStemmingNode,{'label':'avg_price_hotel','stems':8,'min_width':20,'parents':['clicks']}), 
         (LinearStemmingNode,{'label':'distance_to_center','stems':16,'parents':['clicks']}), 
         (AdaptStemmingNode,{'label':'rating','stems':8,'parents':['clicks']}),
         (CategoricalNode,{'label':'city_id','parents':['clicks']}),
         (CategoricalNode,{'label':'stars','ids':[0,1,2,3,4,5],'parents':['clicks']}),
         (AdaptStemmingNode,{'label':'clicks','stems':96,'min_width':5})]


# %% load data
inpath, outpath = filter_argv(*sys.argv)

cols = np.arange(1,11)
dtypes = [("hotel_id",np.uint64),("city_id",np.uint64),("clicks",float),
         ("stars",float),("distance_to_center",float),("avg_price_hotel",float),
         ("rating",float),("nmbr_partners_index",float),
         ("avg_rel_saving",float),("avg_rank",float)]

print("Load data from csv file '%s'..." % inpath)
data = np.genfromtxt(inpath,delimiter=',',dtype=dtypes,usecols=cols,names=True)
print("finished")

missing = np.isnan(data['clicks'])

known  = data[~missing]
unknown = data[missing]

# split for own validation
HOLD_OUT = 50000
rndset = rand(len(known)).argsort()
test = known[rndset[:HOLD_OUT]]
train = known[rndset[HOLD_OUT:]]


# %% train full model and save prediction to csv file
print('Train models with all data and without the holdout test samples...')
## Uncomment to train with all samples
#network = BayesNetwork(nodes,parents).learn(data,weighting(data),em_ignore=['city_id'])
network = BayesNetwork(nodes).learn(known,weighting(known))
tnetwork = BayesNetwork(nodes).learn(train,weighting(train))
print('finished')


print('Validate performance of models on hold out data ...')
pred_baseline_train = baseline(train)
perf_baseline_test = performace_measure(test,pred_baseline_train)
print('\tweighted rmse baseline on test data: \t\t%.3f' % perf_baseline_test)
pred_train_test = tnetwork.mean_prediction(test)
perf_train_test = performace_measure(test,pred_train_test)
print('\tweighted rmse using model with holdout: \t%.3f' % perf_train_test)
pred_full_test = network.mean_prediction(test)
perf_full_test = performace_measure(test,pred_full_test)
print('\tweighted rmse using full model: \t\t%.3f' % perf_full_test)
#pred_full_known = network.mean_prediction(known)
#perf_full_known = performace_measure(known,pred_full_known)
#print('\tweighted rmse on known data using full model: \t%.3f' % perf_full_known)
print('finished')


print('Predict unknown clicks...')
prediction = np.zeros(unknown.shape,dtype=[('hotel_id', '<u8'), ('clicks', '<u8')])
prediction['hotel_id'] = unknown['hotel_id']
prediction['clicks'] = network.mean_prediction(unknown)
print('finished')

print("Save prediction to csv file '%s'..." % outpath)
np.savetxt(outpath,prediction,delimiter=',',fmt='%u')
print('finished')

