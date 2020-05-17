import pdb
import numpy as np
from sklearn.metrics import auc
import json
import pandas as pd
from gensim.models.poincare import PoincareModel, PoincareRelations, LexicalEntailmentEvaluation
from gensim.models.poincare import *
from gensim.test.utils import datapath
from gensim.models.poincare import PoincareKeyedVectors


data_dir = '../../../Documents/kg_embeddings_data/'

relations = []
edgelist = pd.read_csv(data_dir+'data/edgelist_has_function_free.txt', header = None, sep = ' ')
edgelist_data = edgelist.values

data = pd.read_csv(data_dir+'data/mapping_RDFGraph.txt', header = None, sep = '\t')
data = data.values
mapping_dict = dict(zip(data[:,1],data[:,0]))


for item in edgelist_data:
	it1 = mapping_dict[item[0]].split('/')[-1]
	it2 = mapping_dict[item[1]].split('/')[-1]
	relations.append((it1,it2))

model = PoincareModel(relations, size=128)
poincare_model = model.kv.load_word2vec_format(data_dir+"data/poincare/embeddings_has_function_poincare_free.txt")


embds_dict = {}
for i in range(len(poincare_model.vocab)):
	embedding_vector = poincare_model[poincare_model.index2entity[i]]
	if embedding_vector is not None:
		embds_dict[poincare_model.index2entity[i]] = embedding_vector


ent1_embeddings = {}
ent2_embeddings = {}
for item in embds_dict:
	ent = item.split('/')[-1]
	if ent.isdigit():
		ent1_embeddings[ent] = np.array(embds_dict[item], dtype = 'float32')
	if ent.startswith('GO_'):
		ent2_embeddings[ent] = np.array(embds_dict[item], dtype = 'float32')

allents = ent2_embeddings.keys()


train_graph = list(open(data_dir+'data/train_has_function.txt').readlines())
train_edges = [item.strip().split() for item in train_graph]
train_nodes = [item[0] for item in train_edges]
train_nodes = [mapping_dict[int(it)].split('/')[-1] for it in train_nodes]
train_edges = [(mapping_dict[int(item[0])].split('/')[-1],mapping_dict[int(item[1])].split('/')[-1]) for item in train_edges]


test_graph = list(open(data_dir+'data/test_has_function.txt').readlines())
test_edges = [item.strip().split() for item in test_graph]
test_edges = [(mapping_dict[int(item[0])].split('/')[-1],mapping_dict[int(item[1])].split('/')[-1]) for item in test_edges]

ranks = []
top10 = []
for item in test_edges:
	ent1 = item[0]
	ent2 = item[1]
	if ent1 in ent1_embeddings and ent2 in ent2_embeddings:
		if ent1 in train_nodes:
			train_ents = [item[1] for item in train_edges if item[0]== ent1]
			to_test = list(set(allents) - set(train_ents))
		else:
			to_test = list(allents)
		y_pred = poincare_model.distances(ent1, to_test) #poincare distances
		sorted_idx = np.argsort(y_pred)
		sort_entities = [to_test[arg] for arg in sorted_idx]
		idx = sort_entities.index(ent2)
		ranks.append(idx)
		if idx <= 10:
			top10.append(1)

print('has_function_poincare_free')
print('mean ranks: {}'.format(np.mean(ranks)))
topat10 = float(np.sum(top10)/float(len(ranks)))*100
print('%the top @10: {}'.format(topat10))

pdb.set_trace()


