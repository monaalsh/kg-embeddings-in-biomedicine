import pdb
import json
import pandas as pd
import numpy as np

data_dir = '../../../Documents/kg_embeddings_data/data/rescal/'

with open(data_dir+'has_target_free/rescal_embedding.vec.json') as f:
	embeddings = json.load(f)

embds_list = embeddings['ent_embeddings']
rel_list = embeddings['rel_matrices']

mapping = pd.read_csv(data_dir+'has_target_free/entity2id.txt', header = None, sep = '\t', skiprows=1)
mapping_data = mapping.values
entities = mapping_data[:,0]

mapping = pd.read_csv(data_dir+'has_target_free/relation2id.txt', header = None, sep = '\t', skiprows=1)
mapping_data = mapping.values
relations = mapping_data[:,0]

mapping2 = pd.read_csv(data_dir+'mapping_RDFGraph.txt', header=None, sep = '\t')
mapping2_data = mapping2.values
mapping2_dict = dict(zip(mapping2_data[:,1],mapping2_data[:,0]))


train_graph = list(open(data_dir+'train_has_target.txt').readlines())
train_edges = [item.strip().split() for item in train_graph]
train_nodes = [item[0] for item in train_edges]
train_nodes = [mapping2_dict[int(it)].split('/')[-1] for it in train_nodes]
train_edges = [(mapping2_dict[int(item[0])].split('/')[-1],mapping2_dict[int(item[1])].split('/')[-1]) for item in train_edges]


test_graph = list(open(data_dir+'test_has_target.txt').readlines())
test_edges = [item.strip().split() for item in test_graph]
test_edges = [(mapping2_dict[int(item[0])].split('/')[-1],mapping2_dict[int(item[1])].split('/')[-1]) for item in test_edges]




ent1_embeddings = {}
ent2_embeddings = {}
embds_dict = dict()
rel_matrices = dict()

for i, item in enumerate(entities):
	ent = mapping2_dict[item].split('/')[-1]
	embds_dict[ent] = embds_list[i]

for i, item in enumerate(relations):
	rel = mapping2_dict[item].split('/')[-1]
	rel_matrices[rel] = np.array(rel_list[i]).reshape(128,128)

for item in embds_dict:
	if item.startswith('CID'):
		ent1_embeddings[item] = np.array(embds_dict[item], dtype = 'float32')
	if item.isdigit():
		ent2_embeddings[item] = np.array(embds_dict[item], dtype = 'float32')
	else:
		continue

allents = ent2_embeddings.keys()
# r_mat = rel_matrices['has_target']

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
            # ent1_rr = np.matmul(ent1_embd, r_mat)
            # score =  np.matmul(ent1_rr, ent2_embd)
            score =  np.matmul(ent1_embd, ent2_embd) #compute this in free setup, no relation matrix
            y_pred.append(score)

        sorted_idx = np.argsort(y_pred)[::-1]
        sort_entities = [to_test[arg] for arg in sorted_idx]
        idx = sort_entities.index(ent2)
        ranks.append(idx)
        if idx <= 10:
            top10.append(1)

print('has_target_rescal_free')
print('mean ranks: {}'.format(np.mean(ranks)))
topat10 = float(np.sum(top10)/float(len(ranks)))*100
print('%the top @10: {}'.format(topat10))

pdb.set_trace()
