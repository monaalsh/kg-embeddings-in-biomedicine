import pdb
import json
import itertools
import numpy as np
import pandas as pd


data_dir = '../../../Documents/kg_embeddings_data/'

# use skiprows with walking RDFOWL embeddings and poincare embeddings only
data = pd.read_csv(data_dir+'data/transE/embeddings_has_sideeffect_transE.txt', header = None, sep = ' ')
embds_data = data.values
embds_dict = dict(zip(embds_data[:,0],embds_data[:,1:]))

data = pd.read_csv(data_dir+'data/mapping_RDFGraph.txt', header = None, sep = '\t')
data = data.values
mapping_dict = dict(zip(data[:,1],data[:,0]))


ent1_embeddings = {}
ent2_embeddings = {}

for item in embds_dict:
        ent = item.split('/')[-1]
        if ent.startswith('CID'):
            ent1_embeddings[ent] = np.array(embds_dict[item], dtype = 'float32')
        if ent.startswith('HP_'):
            ent2_embeddings[ent] = np.array(embds_dict[item], dtype = 'float32')
        else:
            continue


allents = ent2_embeddings.keys()
# in the Free setup the relation embeddings can't be generated
rel_embd = np.array(embds_dict['http://bio2vec.net/relation/has_sideeffect'])

# pdb.set_trace()

train_graph = list(open(data_dir+'data/train_has_sideeffect.txt').readlines())
train_edges = [item.strip().split() for item in train_graph]
train_nodes = [item[0] for item in train_edges]
train_nodes = [mapping_dict[int(it)].split('/')[-1] for it in train_nodes]
train_edges = [(mapping_dict[int(item[0])].split('/')[-1],mapping_dict[int(item[1])].split('/')[-1]) for item in train_edges]


test_graph = list(open(data_dir+'data/test_has_sideeffect.txt').readlines())
test_edges = [item.strip().split() for item in test_graph]
test_edges = [(mapping_dict[int(item[0])].split('/')[-1],mapping_dict[int(item[1])].split('/')[-1]) for item in test_edges]


ranks = []
top10 = []
#report mean rank and top @10 as in kg methods evaluation
for item in test_edges:
    ent1 = item[0]
    ent2 = item[1]
    if ent1 in ent1_embeddings and ent2 in ent2_embeddings:
        if ent1 in train_nodes:
            train_ents = [item[1] for item in train_edges if item[0]== ent1]
            to_test = list(set(allents) - set(train_ents))
        else:
            to_test = list(allents)

        y_pred = []
        for ii in to_test:
            ent1_embd = ent1_embeddings[ent1]
            ent2_embd = ent2_embeddings[ii]
            score = np.linalg.norm(((ent1_embd+rel_embd) - ent2_embd),1) 
            # score = np.linalg.norm((ent1_embd - ent2_embd),1) # in Free setup
            y_pred.append(score)

        sorted_idx = np.argsort(y_pred)
        sort_entities = [to_test[arg] for arg in sorted_idx]
        idx = sort_entities.index(ent2)
        ranks.append(idx)
        if idx <= 10:
            top10.append(1)

print('has_sideeffect_transE')
print('mean ranks: {}'.format(np.mean(ranks)))
topat10 = float(np.sum(top10)/float(len(ranks)))*100
print('%the top @10: {}'.format(topat10))
pdb.set_trace()