import pdb
import json
import pandas as pd




data_dir = '../../Documents/kg_embeddings_data/data/'

with open(data_dir+'rescal/has_indication/rescal_embedding.vec.json') as f:
	embeddings = json.load(f)

embds_list = embeddings['ent_embeddings']
# rel_list = embeddings['rel_matrices'] #comment this for the free setup

mapping = pd.read_csv(data_dir+'rescal/has_indication/entity2id.txt', header = None, sep = '\t', skiprows=1)
mapping_data = mapping.values
entities = mapping_data[:,0]

mapping = pd.read_csv(data_dir+'rescal/has_indication/relation2id.txt', header = None, sep = '\t', skiprows=1)
mapping_data = mapping.values
relations = mapping_data[:,0]

mapping2 = pd.read_csv(data_dir+'mapping_RDFGraph.txt', header=None, sep = '\t')
mapping2_data = mapping2.values
mapping2_dict = dict(zip(mapping2_data[:,1],mapping2_data[:,0]))


embds_dict = dict()

for i, item in enumerate(entities):
	ent = mapping2_dict[item].split('/')[-1]
	embds_dict[ent] = embds_list[i]

with open(data_dir+'rescal/has_indication/embeddings_has_indication_rescal.txt','w') as f1:
	for item in embds_dict:
		vec = embds_dict[item]
		vec = [str(it) for it in vec]
		f1.write(item+' '+' '.join(vec)+'\n')

# pdb.set_trace()
