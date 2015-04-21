# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:06:44 2015

Notes
------

Defines a simple framework for bayes inference in a bayes network.
The central class is `BayesNetwork`, which provides all methods to learn 
the network parameters from provided data and compute posterior values and
mean estimated for continuous variables. 

The Framework defines a number of nodes which can be used to describe the data

- `CategoricalNode` - A node for discrete categorical data.

- `LinearStemmingNode` - Uses linearly spaced buckets to descretize the \
    continuous data.

- `AdaptStemmingNode` - Uses an data dependent mapping function to \
    descretize continuous data.

- `StemmingNode` - Base class for all continuous data approximated by a \
    multinomial distribution using. The bucket boundaries have to be defined \
    by the user.
    
    
@author: Fabian Schmieder
"""
import numpy as np
import sys
from numpy.random import rand

class BayesError(Exception):
    pass

def build_slice(ind):
    return slice(None) if np.isnan(ind) else slice(int(ind),int(ind)+1)
build_slice_arr = np.frompyfunc(build_slice,1,1)

class Node(object):
    """
    Abstract base class for all bayes nodes.
    """
    def __init__(self,label,points,parents=None,network=None,dim=0):
        self.label = label
        self.points = points
        self.ptable = None
        self.samples = 0.0
        self.dim = dim
        self.parents = parents or []
        self.network = network
        
    def parameters(self):
        return {'label':self.label,'points':self.points,'parents':self.parents}
        
    def prepare(self,data,weights):
        self.samples = 0.0
        self.ptable = None
        return self
        
    def set_connections(self,connection):
        network = self.network
        parents = [network.nodes[network.labelmap[p]] for p in self.parents]
        parents_dim = [p.dim for p in parents]
        self.dims = np.sort(np.r_[self.dim,parents_dim].astype(int))
        self.connection = connection
        
    def set_ptable(self,cnts):
        self.samples = cnts.sum()
        self.marginal = np.add.reduce(cnts,self.dim,keepdims=True)
        self.ptable = cnts/self.marginal
        self.marginal /= self.samples
        self.tmpslice = np.array([slice(None) for d in cnts.shape])
        
    def update_ptable(self,cnts):
        cnts += self.samples*self.ptable*self.marginal
        self.samples = cnts.sum()
        self.marginal = np.add.reduce(cnts,self.dim,keepdims=True)
        self.ptable = cnts/self.marginal
        self.marginal /= self.samples
        
    def fill_factor(self,smplind):
        self.tmpslice[self.dims] = smplind[self.dims]
        self.connection.factor = Factor(self.ptable[tuple(self.tmpslice)],
                                        self.dim)
        
    def map_data(self,data):
        raise RuntimeError('Abstract Node base class')

class CategoricalNode(Node):
    """
    Defines a node for categorical data following a multinomial distribution.
    
    Parameters
    ----------
    label : string
        The identifier for the node. This has to correspond to a column in the
        structured numpy array containing the data for the network.
    ids : list,np.ndarray,optional
        If provided defines the allowed values for this node. If `None` the
        ids will be extracted from the provided data before 
        training, default `None`
    parents : list,optional
        A list containing the labels of all parent nodes of this node, 
        default `[]`.
    """
    def __init__(self,label,ids=None,*args,**kwargs):
        points = np.array([]) if ids is None else np.asarray(ids)
        super(CategoricalNode,self).__init__(label,points,*args,**kwargs)
        self.ids = list(ids) if ids is not None else []
        
    def prepare(self,data,weighting):
        x = data[self.label]
        active = ~np.isnan(x)
        self.points = np.asarray(self.ids or np.unique(x[active]))
        return super(CategoricalNode,self).prepare(data,weighting)
        
    def map_data(self,data):
        x = data[self.label]
        points = self.points
        idx_pos = np.arange(points.size,dtype=float)
        idx = np.interp(x,points,idx_pos,left=np.NaN,right=np.NaN)
        idx[(idx%1) != 0] = np.NaN
        return idx

    def parameters(self):
        d = super(CategoricalNode,self).parameters()
        del d['points']
        for key in {'ids'}:
            d[key] = getattr(self,key)
        return d
        
class StemmingNode(Node):
    """
    Defines a node for continuous data. The data will be approximated with
    a multinomial distribution by binning the data to buckets using the 
    provided bucket boundaries.
    
    Parameters
    ----------
    label : string
        The identifier for the node. This has to correspond to a column in the
        structured numpy array containing the data for the network.
    bounds : list,np.ndarray
        The boundaries of the buckets to use for binning the data to a set
        of discrete values. The number of buckets is one less than the number
        of boundary values, i.e. the length of `bound`.
    cutoff : bool,optional
        Defines how to handle values outside of the defined bucket boundaries.
        If `True` values outside the bucket boundaries will be set to NaN and
        treated as a missing value. If `False` the closest bucket will be used
        for samples outside the boundaries, default `False`.
    parents : list,optional
        A list containing the labels of all parent nodes of this node, 
        default `[]`.
    """
    def __init__(self,label,bounds,cutoff=False,*args,**kwargs):
        bounds = np.atleast_1d(bounds)
        super(StemmingNode,self).__init__(label,.5*(bounds[1:]+bounds[:-1]),
                                               *args,**kwargs)
        self.bounds = bounds
        self.idxmap = np.arange(bounds.size,dtype=float)-.5
        self.cutoff = True
        
    def set_bounds(self,bounds):
        bounds = np.atleast_1d(bounds)
        self.idxmap = np.arange(bounds.size,dtype=float)-.5
        self.bounds = bounds
        self.points = .5*(bounds[1:]+bounds[:-1])
        return self
        
    def parameters(self):
        d = super(StemmingNode,self).parameters()
        del d['ids']
        for key in {'bounds','cutoff'}:
            d[key] = getattr(self,key)
        return d
        
    def map_data(self,data):
        bounds = np.NaN if self.cutoff else None
        return np.interp(data[self.label],self.bounds,self.idxmap,left=bounds,
                         right=bounds)
        
    def prepare(self,data,weights):
        x = data[self.label]
        idx = self.map_data(data)
        active = ~np.isnan(idx)
        x = x[active]
        idx = idx[active].astype(int)
        points = np.zeros(self.points.shape)
        if weights is None:
            for cid in range(points.size):
                points[cid] = x[idx==cid].mean()
        else:
            weights = weights[active]
            for cid in range(points.size):
                ws = weights[idx==cid]
                points[cid] = np.inner(x[idx==cid],ws)/ws.sum()
                
        mpoints = .5*(self.bounds[1:]-self.bounds[:-1])
        self.points = points
        self.points[np.isnan(points)] = mpoints[np.isnan(points)]
        return super(StemmingNode,self).prepare(data,weights)

class LinearStemmingNode(StemmingNode):
    """
    A special node for continuous data. This subclass of `StemmingNode` 
    defines the boundaries of the buckets by linarly spacing them between
    to minimal and maximal values in the provided training data.
    
    Parameters
    ----------
    label : string
        The identifier for the node. This has to correspond to a column in the
        structured numpy array containing the data for the network.
    stems : integer
        The number of buckets to use. The resulting boundaries will be linarly
        spaced between the 2th and 98th percentiles of the provided training
        values.
    cutoff : bool,optional
        Defines how to handle values outside of the defined bucket boundaries.
        If `True` values outside the bucket boundaries will be set to NaN and
        treated as a missing value. If `False` the closest bucket will be used
        for samples outside the boundaries, default `False`.
    parents : list,optional
        A list containing the labels of all parent nodes of this node, 
        default `[]`.
    """
    def __init__(self,label,stems,*args,**kwargs):
        super(LinearStemmingNode,self).__init__(label,np.arange(stems+1),
                                                    *args,**kwargs)
        self.stems = stems
        
    def prepare(self,data,weights):
        x = data[self.label]
        active = ~np.isnan(x)
        low,high = np.percentile(x[active],[1,99])
        bounds = np.linspace(low,high,self.stems+1)
        self.set_bounds(bounds)
        return super(LinearStemmingNode,self).prepare(data,weights)

    def parameters(self):
        d = super(LinearStemmingNode,self).parameters()
        del d['bounds']
        for key in {'stems'}:
            d[key] = getattr(self,key)
        return d

class AdaptStemmingNode(StemmingNode):
    """
    A special node for continuous data. This subclass of `StemmingNode`
    used a data dependend mapping to define the boundaries of the buckets. 
    The adaptive mapping is based on a simple approximation of the 
    inverse cumulative distribution function of the provided training data.
    
    Parameters
    ----------
    label : string
        The identifier for the node. This has to correspond to a column in the
        structured numpy array containing the data for the network.
    stems : integer
        The number of buckets to use.
    min_width : float,optional
        Defines the minimal distance between bucket boundaries, default `0.0`.
    cutoff : bool,optional
        Defines how to handle values outside of the defined bucket boundaries.
        If `True` values outside the bucket boundaries will be set to NaN and
        treated as a missing value. If `False` the closest bucket will be used
        for samples outside the boundaries, default `False`.
    parents : list,optional
        A list containing the labels of all parent nodes of this node, 
        default `[]`.
        
    Notes
    -----
    By using this mapping the resulting marginal multinomial distribution 
    will have close to equal probabilities for the individual buckets. The
    resulting boundaries will be for away in regions with low probability mass
    and closly space in regions with high probability mass.
    
    """
    def __init__(self,label,stems,min_width=0.0,*args,**kwargs):
        super(AdaptStemmingNode,self).__init__(label,np.arange(stems+1),
                                                    *args,**kwargs)
        self.min_width = min_width
        self.stems = stems
        
    def prepare(self,data,weights):
        stems = self.stems
        min_width = self.min_width
        x = data[self.label]
        active = ~np.isnan(x)
            
        length = 100.0/stems
        breaks = np.linspace(0.0,100.0,stems+1)
        x_perc = np.percentile(x[active],breaks)
        x_perc[0] -= 1e-6*(x_perc[-1]-x_perc[0])
        if min_width:
            x_opt = x_perc.copy()
            for i in range(stems):
                if (x_opt[i+1]-x_opt[i]) < min_width:
                    x_opt[i+1] = x_opt[i]+min_width
            perc_opt = np.interp(x_opt,x_perc,breaks)
            for i in range(stems):
                nval = perc_opt[i]+length*(100-perc_opt[i])/(100-breaks[i])
                if perc_opt[i+1] < nval:
                    perc_opt[i+1] = nval
            perc_opt[-1] = 100.0
            x_perc = np.percentile(x[active],perc_opt)
            x_perc[0] -= 1e-6*(x_perc[-1]-x_perc[0])
            
        self.set_bounds(x_perc)
        return super(AdaptStemmingNode,self).prepare(data,weights)

    def parameters(self):
        d = super(AdaptStemmingNode,self).parameters()
        del d['bounds']
        for key in {'stems','min_width'}:
            d[key] = getattr(self,key)
        return d
        
class Factor(object):
    def __init__(self,ptable,dim):
        self.ptable = ptable
        self.id = dim

class Connection(object):
    def __init__(self,dim):
        self.factor = None
        self.dim = dim

class BayesNetwork(object):
    """
    Defines a Bayes Network and provides methods to learn its probabilities 
    from the provided data and inference methods to compute posterior 
    probabilities.
    
    Parameters
    -----------
    nodes : list
        A list defining all nodes in the network. Each element in the list
        is a tuple `(nodeclass,nodeparam)` defining the class and parameters
        of the node, see each node for a list of parameters. 
        
    Examples
    --------
    Create a simple network containing two categorical nodes `A` and `B`.
    
    >>> network = BayesNetwork([CategoricalNode,{'label':'A'},
                                CategoricalNode,{'label':'B','parents':['A']}])
                                
    Define a structured numpy array to hold 100 data samples
    
    >>> data = np.zeros(100,dtype=[('A', 'f8'), ('B', 'f8')])
    
    Learn parameters of the network from data
    
    >>> network.learn(data)
    
    Compute the a'posterior pobabilities of node `A` for the given data
    
    >>> network.posterior(data,'A')
    
    Notes
    ------
    The the order the factorials are reduced for the inference is given by
    the order of the provided `nodes` list when creating the network. 
    Therefore. the order can be important for a fast execution. As a general 
    rule the nodes many children should be placed at the end of the list.
    """
    def __init__(self,nodes):
        self.nodes = [cls(dim=i,network=self,**p) for i,(cls,p) in enumerate(nodes)]
        self.shape = tuple(n.points.size for n in self.nodes)
        self.labels = np.array([n.label for n in self.nodes])
        self.labelmap = dict((l,i) for i,l in enumerate(self.labels))
        self.parents = []
        for i,node in enumerate(self.nodes):
            try:
                self.parents.append([self.labelmap[p] for p in node.parents])
            except KeyError as err:
                raise BayesError("Could not find node '%s' while building connections" % err)
            
        self.children = [[] for n in nodes]
        self.roots = [i for i,p in enumerate(self.parents) if not p]
        self.conns = [[Connection(i)] for i in range(len(nodes))]
        for i,(node,conns) in enumerate(zip(self.nodes,self.conns)):
            node.set_connections(conns[0])
            for p in self.parents[i]:
                self.conns[p].append(conns[0])
                self.children[p].append(i)
        
    def learn(self,data,weights=None,niter=10,em_ignore=None,
              ghost_samples=1e-5,batch=1000,file=sys.stdout):
        """
        Learn the network parameters from the provided data.
        
        Parameters
        ---------
        data : structured numpy.ndarray
            A structured numpy array containing the data to use.
        weights : np.ndarray,optional
            If provided defines a custom weight for each sample when
            estimating the probabilities, default equal weights for all
            samples.
        niter : integer,optional
            The number of EM iterations to use, default `5`.
        em_ignore : list,optional
            If provided can be used to exclude specific nodes from the
            EM updates to reduce the memory or time requirements for the 
            training, default `[]`.
        ghost_samples : float,optional
            Is used to ensure nonzero probabilities, default `1e-5'.
        batch : integer,optional
            Defnies the size of the batch used in the EM iterations inner 
            loop to reduce the required memory for large datasets.
        file : stream,optional
            Will be passed to the print calls, can be used to redirect
            to status output, default `sys.stdout`.
            
        Returns
        -------
        self : BayesNetwork
            Returns a reference to itself
            
        """
        em_ignore = set(self.labelmap[n] for n in (em_ignore or []))
        for node in self.nodes:
            node.prepare(data,weights)
        shape = np.array([n.points.size for n in self.nodes])
        
        ids = np.zeros((len(self.nodes),len(data)),dtype=np.object)
        nslice = slice(None)
        for i,node in enumerate(self.nodes):
            cids = node.map_data(data)
            active = ~np.isnan(cids)
            ids[i,~active] = nslice
            ids[i,active] = cids[active].astype(int)
        
        nullcnts = []
        for i,node in enumerate(self.nodes):
            inds = np.sort(np.r_[i,self.parents[i]].astype(int))
            cshape = shape[inds]
            multiplier = np.r_[1,cshape[::-1]].cumprod()[::-1]
            maxind, multiplier = multiplier[0], multiplier[1:]
            cids = ids[inds]
            allactive = ~np.any(cids==nslice,0)
            cids = np.dot(multiplier,cids[:,allactive].astype(int))
            
            if cids.size == 0:
                cnts = rand(maxind)
                cnts /= cnts.sum()
            elif weights is None:
                cnts = np.bincount(cids,minlength=maxind)
            elif np.sum(weights[allactive]) == 0:
                cnts = rand(maxind)
                cnts /= cnts.sum()
            else:
                cnts = np.bincount(cids,minlength=maxind,
                                   weights=weights[allactive])
                
            cnts.shape = tuple(cshape)
            cnts = np.asarray(cnts,dtype=float)
                
            cnts += ghost_samples/shape[i]
            
            fullshape = np.ones(shape.shape,dtype=int)
            fullshape[inds] = cshape
            cnts.shape = tuple(fullshape)
            nullcnts.append(cnts)
            node.set_ptable(cnts)
            
        self.shape = shape
        if not niter:
            return self
            
        toupdate = []
        neadedpost = set()
        neadedparents = set()
        for i,node in enumerate(self.nodes):
            # only probabilities with missing parents have to be updated
            missing_parents = np.any(ids[self.parents[i],:]==nslice)
            known_children = np.any(ids[i,:]==nslice) and self.children[i]
            if (missing_parents or known_children) and i not in em_ignore:
                inds = [i]+self.parents[i]
                shapes = []
                nshape = np.ones(len(self.nodes),dtype=int)
                for j in inds:
                    cshape = np.ones(len(self.nodes)+1,dtype=int)
                    cshape[-1] = batch
                    nshape[j] = cshape[j] = shape[j]
                    shapes.append(cshape)
                cnts = np.zeros(nshape)
                neadedpost.update(inds)
                neadedparents.update(self.parents[i])
                toupdate.append((node,cnts,inds,shapes))
                
        if toupdate and file:
            print('Start EM Algorithm for missing data inference',file=file)
            print('\tUpdating: %s' % ','.join(tpl[0].label for tpl in toupdate),file=file)
            print('\tNeeded posterior: %s' % ','.join(self.labels[dim] for dim in neadedpost),file=file)
        elif not toupdate:
            # no updates required
            return self

        active = np.any(ids[list(neadedparents)]==nslice,0)
        posterior = [None]*len(shape)
        constcnts = [None]*len(shape)
        for dim in neadedpost:
            # compute the cnts for the samples with perfect information
            # as they will not change in the EM iterations
            inds = np.sort(np.r_[dim,self.parents[dim]].astype(int))
            cshape = shape[inds]
            fullshape = np.ones(shape.shape,dtype=int)
            fullshape[inds] = cshape
            
            
            multiplier = np.r_[1,cshape[::-1]].cumprod()[::-1]
            maxind, multiplier = multiplier[0], multiplier[1:]
            cids = ids[inds]
            allactive = ~np.any(cids==nslice,0)
            cids = np.dot(multiplier,cids[:,allactive].astype(int))
            if weights is None:
                cnts = np.bincount(cids,minlength=maxind)
            else:
                cnts = np.bincount(cids,minlength=maxind,
                                   weights=weights[allactive])
            cnts = np.asarray(cnts,dtype=float).reshape(fullshape)
            cnts += ghost_samples/shape[dim]
            constcnts[dim] = cnts
            posterior[dim] = np.zeros((shape[dim],batch))
        
        
        data = data[active]
        ids = ids[:,active]
        if weights is not None:
            weights = weights[active]
            
        # precompute the index array required by posterior
        postids = np.zeros((len(self.nodes),len(data)),dtype=float)
        for i,node in enumerate(self.nodes):
            postids[i] = node.map_data(data)
        postids = build_slice_arr(postids.T)
        
        sampleind = np.arange(len(data))
        wsub = np.zeros(batch)
        for citer in range(niter):
            for s in range(0,len(data),batch):
                subids = ids[:,s:s+batch]
                sind = sampleind[s:s+batch]
                if file:
                    print('iter=%d of %d, sample=%d of %d' % (citer+1,niter,
                                                              s,len(data)),
                                                              file=file)
                # compute posterior of known and unkown values
                for dim in neadedpost:
                    cpost = posterior[dim]
                    missing = (subids[dim]==nslice)
                    valid_smpl = np.flatnonzero(~missing)
                    valid_ids = subids[dim,valid_smpl].astype(int)
                    cpost[:] = 0.0
                    cpost[valid_ids,valid_smpl] += 1.0
                    cpost[:,missing] = self.posterior(data[sind[missing]],
                                                      self.labels[dim],
                                                      postids[sind[missing]]).T
                    
                if weights is not None:
                    wsub[:sind.size] = weights[sind]

                # update counts
                for node,cnts,inds,shapes in toupdate:
                    curcnts = posterior[inds[0]].reshape(shapes[0])
                    for i,cshape in zip(inds[1:],shapes[1:]):
                        curcnts = curcnts*posterior[i].reshape(cshape)
                    if weights is None:
                        cnts += curcnts.sum(-1)
                    else:
                        cnts += np.inner(curcnts,wsub)
                    
            # set new ptables
            for node,cnts,inds,shapes in toupdate:
                cnts += constcnts[node.dim]
                node.set_ptable(cnts)
                cnts[:] = 0.0
                
        return self
        
    def mean_prediction(self,data,node=None):
        """
        Computes the mean prediction based on the infered posterior 
        probabilities of the node given the provided data.
        
        Parameters
        ----------
        data : structured numpy.ndarray
            A structured numpy array containing the data to use.
        node : string, optional
            Defines the node which should be predicted, default is the last 
            node in the network.
            
        Returns
        --------
        pred : np.ndarray
            A numpy array of the same shape as the provided data containing
            the computed mean values.
        """
        try:
            postind = self.labelmap[node or self.labels[-1]]
        except KeyError as err:
            raise BayesError("Could not find node '%s'" % err)
        posterior = self.posterior(data,node)
        return np.dot(posterior,self.nodes[postind].points)
    
    def posterior(self,data,node=None,ids=None):
        """
        Computes the posterior probabilities of a node given the provided 
        data or the likelihood of the data for the network.
        
        Parameters
        ----------
        data : structured numpy.ndarray
            A structured numpy array containing the data to use.
        node : string, optional
            Defines the node which should be predicted. If node is `None` no 
            node will be used and the likelihood of the sample will 
            be returned.
            
        Returns
        --------
        posterior : np.ndarray
            A numpy array containing the computed posterior probabilities.
        """
        if node is not None:
            try:
                postind = self.labelmap[node]
            except KeyError as err:
                raise BayesError("Could not find node '%s'" % err)
        else:
            postind = -1
        
        if ids is None:
            ids = np.zeros((len(self.nodes),len(data)),dtype=float)
            for i,node in enumerate(self.nodes):
                ids[i] = node.map_data(data)
            if postind >= 0:
                ids[postind,:] = np.NaN
            ids = build_slice_arr(ids.T)
        else:
            ids = ids.copy()
            if postind >= 0:
                ids[:,postind] = slice(None)
            
        
        # define hellper variable to check if any factors remain to be reduced
        testind = [slice(None) for n in self.nodes]
        if postind >= 0:
            testind[postind] = 0
            posterior = np.zeros((data.size,self.shape[postind]))
        else:
            posterior = np.zeros(data.size)
        testind = tuple(testind)
        postconns = self.conns[postind]
        for i,cind in enumerate(ids):
            for node in self.nodes:
                node.fill_factor(cind)
            unused = np.ones((len(self.nodes),len(self.nodes)),dtype=bool)
                
            if not all([c.factor.ptable[testind].size==1 for c in postconns]):
                for j,conns in enumerate(self.conns):
                    if j==postind:
                        continue
                    
                    factor = conns[0].factor
                    cunused = unused[j]
                    cunused[factor.id] = False
                    for conn in conns[1:]:
                        if cunused[conn.factor.id]:
                            factor.ptable = factor.ptable*conn.factor.ptable
                            cunused[conn.factor.id] = False
                    factor.ptable = np.add.reduce(factor.ptable,j,keepdims=True)
                    for conn in conns[1:]:
                        conn.factor = factor
            
            if postind < 0:
                posterior[i] = postconns[0].factor.ptable.squeeze()
            else:
                cunused = unused[postind]
                cunused[postconns[0].factor.id] = False
                ptable = postconns[0].factor.ptable
                for conn in postconns[1:]:
                    if cunused[conn.factor.id]:
                        ptable = ptable*conn.factor.ptable
                        cunused[conn.factor.id] = False
                posterior[i] = ptable.ravel()/ptable.sum()
            
        return posterior

    def parameters(self):
        """
        Returns
        -------
        nodes : list
            the node list used to build the network
        """
        return [(type(n),n.parameters()) for n in self.nodes]
