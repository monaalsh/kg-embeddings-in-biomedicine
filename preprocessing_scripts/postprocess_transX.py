import pdb
import numpy as np


entities = []
vecs = []
entities2vec = {}
graph_map = {}

data_dir = '../../../Documents/kg_embeddings_data/'
with open(data_dir+'data/mapping_RDFGraph.txt') as f:
	for line in f:
		items = line.strip().split('\t')
		node = items[0]
		nodetype = items[0].split('/')[-2]
		ID = items[1]
		graph_map[ID] = node


with open(data_dir+'data/entity2id_has_indication_transX.txt') as f:
	for line in f:
		items = line.strip().split('\t')
		entity = items[0]
		entities.append(entity)

with open(data_dir+'data/transE/entity2vec_has_indication_transE.bern') as f:
	for line in f:
		items = line.strip().split('\t')
		vecs.append(items)


for i,item in enumerate(entities):
	entities2vec[item] = vecs[i]

entities1 = []
vecs1 = []
entities2vec1 = {}

with open(data_dir+'data/relation2id_has_indication_transX.txt') as f:
	for line in f:
		items = line.strip().split('\t')
		entity = items[0]
		entities1.append(entity)

with open(data_dir+'data/transE/relation2vec_has_indication_transE.bern') as f:
	for line in f:
		items = line.strip().split('\t')
		vecs1.append(items)


for i,item in enumerate(entities1):
	entities2vec1[item] = vecs1[i]


file1 = open(data_dir+'data/transE/embeddings_has_indication_transE.txt', 'w')

for item in entities2vec:
	if item in graph_map:
		node_IRI = graph_map[item]
		if 'has_' in node_IRI:
			continue
		else:
			file1.write(node_IRI +' '+' '.join(entities2vec[item]) + '\n')

for item in entities2vec1:
	if item in graph_map:
		node_IRI = graph_map[item]
		file1.write(node_IRI +' '+' '.join(entities2vec1[item]) + '\n')


file1.close()

pdb.set_trace()

