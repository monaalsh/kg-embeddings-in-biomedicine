import pdb




data_dir = '../../../Documents/kg_embeddings_data/data/'
entites = set()
relations = set()

with open(data_dir+'edgelist_has_indication.txt') as f:
	for line in f:
		items = line.strip().split()
		entites.add(items[0])
		entites.add(items[1])
		relations.add(items[2])

with open(data_dir+'entity2id_has_indication.txt','w') as f:
	for i, item in enumerate(entites):
		f.write(item +'\t'+ str(i) + '\n')

with open(data_dir+'relation2id_has_indication.txt','w') as f:
	for i, item in enumerate(relations):
		f.write(item + '\t' + str(i)+'\n')

pdb.set_trace()
		
