# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:06:44 2015

@author: Fabian Schmieder
"""
import numpy as np

class BayesError(Exception):
    pass

def build_slice(ind):
    return slice(None) if np.isnan(ind) else slice(int(ind),int(ind)+1)
build_slice_arr = np.frompyfunc(build_slice,1,1)

class BayesNode(object):
    def __init__(self,label,points,dim):
        self.label = label
        self.points = points
        self.ptable = None
        self.samples = 0.0
        self.dim = dim
        self.parents = []
        
    def parameters(self):
        return {'label':self.label,'points':self.points,
                'weighting':self._weighting}
        
    def prepare(self,data,weighting):
        self.samples = 0.0
        return self
        
    def set_connections(self,parents,factor):
        self.parents = parents
        self.dims = np.sort(np.r_[self.dim,[p.dim for p in parents]].astype(int))
        self.connection = factor
        
    def set_ptable(self,cnts):
        self.samples = cnts.sum()
        self.marginal = np.add.reduce(cnts,self.dim,keepdims=True)
        self.ptable = cnts/self.marginal
        self.marginal /= self.samples
        self.tmpslice = np.array([slice(None) for d in cnts.shape])
        
    def fill_factor(self,smplind):
        self.tmpslice[self.dims] = smplind[self.dims]
        self.connection.factor = Factor(self.ptable[tuple(self.tmpslice)])
        
    def map_data(self,data):
        raise RuntimeError('Abstract BayesNode base class')

class BayesMultinominalNode(BayesNode):
    def __init__(self,label,ids=None,*args,**kwargs):
        points = np.array([]) if ids is None else np.asarray(ids)
        super(BayesMultinominalNode,self).__init__(label,points,*args,**kwargs)
        self.ids = list(ids) if ids is not None else []
        
    def parameters(self):
        d = super(BayesMultinominalNode,self).parameters()
        del d['points']
        for key in {'ids'}:
            d[key] = getattr(self,key)
        return d
        
class BayesCategoricalNode(BayesMultinominalNode):
    def __init__(self,label,ids=None,*args,**kwargs):
        points = np.array([]) if ids is None else np.asarray(ids)
        super(BayesCategoricalNode,self).__init__(label,points,*args,**kwargs)
        self.ids = list(ids) if ids is not None else []
        
    def prepare(self,data,weighting):
        x = data[self.label]
        active = ~np.isnan(x)
        self.points = np.asarray(self.ids or np.unique(x[active]))
        return super(BayesMultinominalNode,self).prepare(data,weighting)
        
    def map_data(self,data):
        x = data[self.label]
        points = self.points
        idx_pos = np.arange(points.size,dtype=float)
        idx = np.interp(x,points,idx_pos,left=np.NaN,right=np.NaN)
        idx[(idx%1) != 0] = np.NaN
        return idx

    def parameters(self):
        d = super(BayesCategoricalNode,self).parameters()
        del d['points']
        for key in {'ids'}:
            d[key] = getattr(self,key)
        return d
        
class BayesStemmingNode(BayesMultinominalNode):
    def __init__(self,label,bounds,cutoff=True,*args,**kwargs):
        bounds = np.atleast_1d(bounds)
        super(BayesStemmingNode,self).__init__(label,.5*(bounds[1:]+bounds[:-1]),
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
        d = super(BayesStemmingNode,self).parameters()
        del d['ids']
        for key in {'bounds','cutoff'}:
            d[key] = getattr(self,key)
        return d
        
    def map_data(self,data):
        return np.interp(data[self.label],self.bounds,self.idxmap)
        
    def prepare(self,data,weighting):
        x = data[self.label]
        idx = self.map_data(data)
        active = ~np.isnan(idx)
        x = x[active]
        idx = idx[active].astype(int)
        points = np.zeros(self.points.shape)
        weights = weighting(data[active])
        if weights is None:
            for cid in range(points.size):
                points[cid] = x[idx==cid].mean()
        else:
            for cid in range(points.size):
                ws = weights[idx==cid]
                points[cid] = np.inner(x[idx==cid],ws)/ws.sum()
        
        self.points = points
        return super(BayesStemmingNode,self).prepare(data,weighting)

class BayesAdaptStemmingNode(BayesStemmingNode):
    def __init__(self,label,stems,min_raise=None,*args,**kwargs):
        super(BayesAdaptStemmingNode,self).__init__(label,np.arange(stems+1),
                                                    *args,**kwargs)
        self.min_raise = min_raise
        self.stems = stems
        
    def prepare(self,data,weighting):
        stems = self.stems
        min_raise = self.min_raise
        x = data[self.label]
        active = ~np.isnan(x)
            
        length = 100.0/stems
        breaks = np.linspace(0.0,100.0,stems+1)
        x_perc = np.percentile(x[active],breaks)
        x_perc[0] -= 1e-6*(x_perc[-1]-x_perc[0])
        if min_raise:
            x_opt = x_perc.copy()
            for i in range(stems):
                if (x_opt[i+1]-x_opt[i]) < min_raise:
                    x_opt[i+1] = x_opt[i]+min_raise
            perc_opt = np.interp(x_opt,x_perc,breaks)
            for i in range(stems):
                nval = perc_opt[i]+length*(100-perc_opt[i])/(100-breaks[i])
                if perc_opt[i+1] < nval:
                    perc_opt[i+1] = nval
            x_perc = np.percentile(x[active],perc_opt)
            x_perc[0] -= 1e-6*(x_perc[-1]-x_perc[0])
            
        self.set_bounds(x_perc)
        return super(BayesAdaptStemmingNode,self).prepare(data,weighting)

    def parameters(self):
        d = super(BayesAdaptStemmingNode,self).parameters()
        del d['bounds']
        for key in {'stems','min_raise'}:
            d[key] = getattr(self,key)
        return d
        

class Factor(object):
    def __init__(self,ptable):
        self.ptable = ptable

class Connection(object):
    def __init__(self):
        self.factor = None

class BayesNetwork(object):
    def __init__(self,nodes,parents):
        self._parents = parents
        self.nodes = [cls(dim=i,**p) for i,(cls,p) in enumerate(nodes)]
        self.shape = tuple(n.points.size for n in self.nodes)
        self.labels = np.array([n.label for n in self.nodes])
        self.labelmap = dict((l,i) for i,l in enumerate(self.labels))
        self.parents = [[]]*len(self.nodes)
        try:
            for label,cpar in parents.items():
                cind = self.labelmap[label]
                self.parents[cind] = [self.labelmap[p] for p in cpar]
        except KeyError as err:
            raise BayesError("Could not find node '%s' while building connections" % err)
            
        self.roots = [i for i,p in enumerate(self.parents) if not p]
        self.conns = [[] for n in self.nodes]
        for i,node in enumerate(self.nodes):
            nconn = Connection()
            cparents = []
            for p in self.parents[i]:
                cparents.append(self.nodes[p])
                self.conns[p].append(nconn)
            node.set_connections(cparents,nconn)
            self.conns[i].append(nconn)
        
    def learn(self,data,ghost_samples=1.0,weighting=None):
        weighting = weighting or (lambda d: None)
        for node in self.nodes:
            node.prepare(data,weighting)
        shape = np.array([n.points.size for n in self.nodes])
        
        ids = np.zeros((len(self.nodes),len(data)),dtype=np.object)
        nslice = slice(None)
        for i,node in enumerate(self.nodes):
            cids = node.map_data(data)
            active = ~np.isnan(cids)
            ids[i,~active] = nslice
            ids[i,active] = cids[active].astype(int)
        
        for i,node in enumerate(self.nodes):
            inds = np.sort(np.r_[i,self.parents[i]].astype(int))
            cshape = shape[inds]
            multiplier = np.r_[1,cshape[::-1]].cumprod()[::-1]
            maxind, multiplier = multiplier[0], multiplier[1:]
            cids = ids[inds]
            allactive = ~np.any(cids==nslice,0)
            anyactive = ~np.all(cids==nslice,0)
            cids = np.dot(multiplier,cids[:,allactive].astype(int))
            
            cnts = np.bincount(cids,minlength=maxind,
                               weights=weighting(data[allactive]))
            cnts.shape = tuple(cshape)
            cnts = np.asarray(cnts,dtype=float)
            cnts += ghost_samples/cnts.size    
            # update with missing data py using the conditional distribution
            # given the known values
            weights = weighting(data[anyactive&~allactive])
            cids = ids[inds][:,anyactive&~allactive]
            if weights is None:
                for idx in zip(*cids):
                    cnts[idx] += cnts[idx]/cnts[idx].sum()
            else:
                for j,idx in enumerate(zip(*cids)):
                    cnts[idx] += weights[j]*cnts[idx]/cnts[idx].sum()
            
            fullshape = np.ones(shape.shape,dtype=int)
            fullshape[inds] = cshape
            cnts.shape = tuple(fullshape)
            node.set_ptable(cnts)

        return self
        
    def mean_prediction(self,data):
        ids = np.zeros((len(self.nodes),len(data)),dtype=float)
        ids[-1,:] = np.NaN
        for i,node in enumerate(self.nodes[:-1]):
            ids[i] = node.map_data(data)
            
        ids = build_slice_arr(ids.T)
        
        pred = np.zeros(data.shape)
        for i,cind in enumerate(ids):
            #cind = np.array([slice(int(v),int(v)+1) if  for v in tpl])
            for node in self.nodes:
                node.fill_factor(cind)
                
            for j,conns in enumerate(self.conns[:-1]):
                factor = conns[0].factor
                for conn in conns[1:]:
                    factor.ptable = factor.ptable*conn.factor.ptable
                factor.ptable = np.add.reduce(factor.ptable,j,keepdims=True)
                for conn in conns[1:]:
                    conn.factor = factor
            
            ptable = self.conns[-1][0].factor.ptable
            for conn in self.conns[-1][1:]:
                ptable *= conn.factor.ptable
            
            pred[i] = np.inner(ptable.ravel(),self.nodes[-1].points)/ptable.sum()
            
        return pred
            

    def parameters(self):
        nodes = [(type(n),n.parameters()) for n in self.nodes]
        labels = self.labels
        parents = dict((l,[labels[p] for p in pl]) for l,pl in zip(labels,self.parents))
        return {'nodes':nodes,'parents':parents}

