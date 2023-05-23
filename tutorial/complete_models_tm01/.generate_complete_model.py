"""
05/23/2023; Tim Marshall; tmchmbusiness@gmail.com

Script to build complete Markov state models for TC5b generated using Folding@Home platform

server@vav22:~/server2/data/PROJ16959/RUN9/CLONE2 - gens 128-571 (640-2860 ns) = 2.22 us Eaxh gen is 5 ns, which snapshots saved ever 100 ps
"""
# standard dependencies
import os, sys, subprocess
# special dependencies
import pyemma
import numpy as np

# specify raw data directory
data_dir = './../.data/'

### BOOLS ###
# save? 
SAVE = True
# random 50% subsample
BOOT = False
# quick, 1 trajectory test run?
QUICK = False

# score features?
# disabled by default due to resource requirements
score_features = False

### LOADER ####

# load in structure
structure = os.path.join(data_dir, 'xtc_atoms.gro')

# load in trajectories
trajectories = [os.path.join(data_dir, s.strip()) for s in """p16959r9c29-whole.xtc 
p16959r9c83-whole.xtc 
p16959r9c100-whole.xtc
p16959r9c109-whole.xtc
p16959r9c127-whole.xtc
p16959r9c151-whole.xtc
p16959r9c166-whole.xtc
p16959r9c219-whole.xtc
p16959r9c312-whole.xtc
p16959r9c394-whole.xtc
p16959r9c631-whole.xtc
p16959r9c726-whole.xtc""".split('\n')]

### SPECIAL CALCULATION CASE ###
# this section yields quick/special models
# use carefully
# if you wish to bootstrap your data
if BOOT:
    import random
    trajectories = random.sample(trajectories, int(len(trajectories)*0.50))
# single trajectory testing
if QUICK:
    trajectories = trajectories[0]

### FEATURIZER ####

# load in structure as pyemma object for selecting features
feat = pyemma.coordinates.featurizer(structure)

# add CA-CA pairwise distances with nearest neighbor exclusion
# this was selected as the best performing feature set based on VAMP-2 scoring
feat.add_distances_ca()
feat_data = pyemma.coordinates.load(trajectories, features=feat)

# SAVE
feat_name_save = f'featurized_data'
if SAVE:
    np.save(feat_name_save, feat_data)

### SCORE FEATRURES ###

# score feature set
validation_fraction = 0.50    #fraction of training/testing split
number_of_splits = 10    #number of times to perform calc, for error purposes
if score_features:
    nval = int(len(feat_data) * validation_fraction)
    scores = np.zeros(number_of_splits)

    for n, _ in enumerate(scores):
        ival = np.random.choice(len(feat_data), size=nval, replace=False)
        valid = [d for i, d in enumerate(feat_data) if i not in ival]
        vamp = pyemma.coordinates.vamp(valid, lag=lag, dim=dim, scaling = 'km')
        scores[n] = vamp.score([d for i, d in enumerate(feat_data) if i in ival])
    # SAVE
    feature_score_name_save = f'feature_scores'
    if SAVE:
        np.save(feature_score_name_save, scores)
print(f'featurization complete')

### TICA ###

# parameters for calculation
dim = 4
tica_lag = 100

#feat_list = np.ndarray.tolist(feat)    #don't worry about this, Tim's paranoia with pyemma
# perform tica
tica = pyemma.coordinates.tica(feat_data, dim=dim, lag=tica_lag, scaling = 'km')

# grab output
tica_getoutput = tica.get_output()

# SAVE
tica_name_save = 'tica_getoutput'
if SAVE:
    np.save(tica_name_save, tica_getoutput)

print(f'tica complete')

### CLUSTERING ###

# parameters for calculation
number_of_clusters = 50
max_iter = 100

# cluster using kmeans algorithm
cluster = pyemma.coordinates.cluster_kmeans(tica_getoutput, k=number_of_clusters, max_iter=max_iter)

cluster_getoutput = cluster.get_output()
cluster_dtrajs = cluster.dtrajs
cluster_centers = cluster.clustercenters

# SAVE
# kmeans is weird in pyemma, so we save a bunch of potentially relevant objects in case of data corruption
discretization_save_name = 'kmeans'
if SAVE:
    cluster_getoutput = cluster.get_output()
    cluster_dtrajs = cluster.dtrajs
    cluster_centers = cluster.clustercenters

    np.save(f'{discretization_save_name}_getoutput.npy', cluster_getoutput)
    np.save(f'{discretization_save_name}_dtrajs.npy', cluster_dtrajs)
    np.save(f'{discretization_save_name}_centers.npy', cluster_centers)

print('discretization complete')

### MSM ###
msm_lag = 200

msm = pyemma.msm.estimate_markov_model(cluster.dtrajs, msm_lag)

# SAVE
# only time it is ok to use pyemma's save function
msm_save_name = 'msm'
if SAVE:
    msm.save(f'{msm_save_name}.h5',  overwrite=True)

print('------------------------------------------------------------------------')
print('MODEL COMPLETE')
print('------------------------------------------------------------------------')

### SCORE MSM ###
number_of_splits = 2
msm_scores_save_name = 'msm_scores'

if SAVE:
    
    scores= [[] for i in range(2)]
    split_scores = []

    for s in range(number_of_splits):
        score = msm.score_cv(cluster.dtrajs, n=1, score_method='VAMP2')
        split_scores.append(score[0])

    kscores_avg = np.average(split_scores)
    kscores_std = np.std(split_scores)

    scores[0].append(kscores_avg)
    scores[1].append(kscores_std)
    
    np.save(msm_scores_save_name, scores)
