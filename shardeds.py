import numpy as np
from hashlib import sha256
import importlib
import json

def sizeOfShard(rseed,per,data, container, shard):
    '''
    Returns the size (in number of points) of the shard before any unlearning request.
    '''
    shards = np.load('containerss/{}/{}/{}/{}/splitfile.npy'.format(per,rseed,data,container), allow_pickle=True)
    
    return shards[shard].shape[0]

def realSizeOfShard(rseed,per,data, container, label, shard):
    '''
    Returns the actual size of the shard (including unlearning requests).
    '''
    shards = np.load('containerss/{}/{}/{}/{}/splitfile.npy'.format(per,rseed,data,container), allow_pickle=True)
    requests = np.load('containerss/{}/{}/{}/{}/requestfile:{}.npy'.format(per,rseed,data, container, label), allow_pickle=True)
    
    return shards[shard].shape[0] - requests[shard].shape[0]

def getShardHash(rseed,per,data, container, label, shard, until=None):
    '''
    Returns a hash of the indices of the points in the shard lower than until
    that are not in the requests (separated by :).
    '''
    shards = np.load('containerss/{}/{}/{}/{}/splitfile.npy'.format(per,rseed,data,container), allow_pickle=True)
    requests = np.load('containerss/{}/{}/{}/{}/requestfile:{}.npy'.format(per,rseed,data, container, label), allow_pickle=True)

    if until == None:
        until = shards[shard].shape[0]
    indices = np.setdiff1d(shards[shard][:until], requests[shard])
#     print('indices', indices, shards)
    string_of_indices = ':'.join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()

def fetchShardBatch(rseed,per,data,container, label, shard, batch_size, dataset, offset=0, until=None):
    '''
    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset
    optionnally located between offset and until (slicing).
    '''
    shards = np.load('containerss/{}/{}/{}/{}/splitfile.npy'.format(per,rseed,data,container), allow_pickle=True)
    requests = np.load('containerss/{}/{}/{}/{}/requestfile:{}.npy'.format(per,rseed,data,container, label), allow_pickle=True)
#     print(until,shard, shards[shard].shape[0])
#     print( shards[0].shape[0], shards[1].shape[0])#, shards[2].shape[0], shards[3].shape[0], shards[4].shape[0])
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))
    if until == None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]
#     print('until' , until)
    limit = offset
#     until = 249215
    while limit <= until - batch_size :
#         print(batch_size, until, limit)
#         if limit + batch_size >= 249215:
#             limit = limit + (249215 -limit)
#         else :
#             limit += batch_size
#         print(limit)
#         if limit > 249215:
#             limit = 249215
        limit += batch_size
        indices = np.setdiff1d(shards[shard][limit-batch_size:limit], requests[shard])
        yield dataloader.load(indices)
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
        yield dataloader.load(indices) 

def fetchTestBatch(dataset, batch_size):
    '''
    Generator returning batches of points from the specified test dataset
    with specified batch_size.
    '''
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    limit = 0
    while limit <= datasetfile['nb_test'] - batch_size:
        limit += batch_size
        yield dataloader.load(np.arange(limit - batch_size, limit), category='test')
    if limit < datasetfile['nb_test']:
        yield dataloader.load(np.arange(limit, datasetfile['nb_test']), category='test')
