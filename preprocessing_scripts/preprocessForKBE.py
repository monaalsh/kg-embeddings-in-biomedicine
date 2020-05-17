import pdb
import json


data_dir = '../../Documents/kg_embeddings_data/data/OpenKE_data/'

entities = list()
IDs = list()
graph_map = dict()

with open(data_dir+'entity2id_has_target_transX_free.txt') as f:
	for line in f:
		items = line.strip().split('\t')
		entities.append(items[0])
		IDs.append(items[1])
		graph_map[items[0]] = items[1]


with open(data_dir+'has_target_free/entity2id.txt','w') as f:
	f.write(str(len(entities))+'\n')
	for ent,iD in zip(entities, IDs):
		f.write(ent+'\t'+iD+'\n')

entities = list()
IDs = list()
with open(data_dir+'relation2id_has_target_transX_free.txt') as f:
	for line in f:
		items = line.strip().split('\t')
		entities.append(items[0])
		IDs.append(items[1])
		graph_map[items[0]] = items[1]

with open(data_dir+'has_target_free/relation2id.txt','w') as f:
	f.write(str(len(entities))+'\n')
	for ent,iD in zip(entities, IDs):
		f.write(ent+'\t'+iD+'\n')


train_triples = list()
with open(data_dir+'edgelist_has_target_free.txt') as f2:
	for line in f2:
		items = line.strip().split()
		train_triples.append(items)

with open(data_dir+'has_target_free/train2id.txt','w') as f1:
	f1.write(str(len(train_triples))+'\n')
	for triple in train_triples:
		f1.write(graph_map[triple[0]]+'\t'+graph_map[triple[1]]+'\t'+graph_map[triple[2]]+'\n')
		

pdb.set_trace()
