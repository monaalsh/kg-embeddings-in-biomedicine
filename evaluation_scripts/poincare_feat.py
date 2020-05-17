import pdb
import numpy as np
from sklearn.metrics import auc
import json
import pandas as pd
from gensim.models.poincare import PoincareModel, PoincareRelations, LexicalEntailmentEvaluation
from gensim.models.poincare import *
from gensim.test.utils import datapath
from gensim.models.poincare import PoincareKeyedVectors


data_dir = '../../../Documents/kg_embeddings_data/data/'
relations = []
edgelist = pd.read_csv(data_dir+'edgelist_has_indication.txt', header = None, sep = ' ')
edgelist_data = edgelist.values

data = pd.read_csv(data_dir+'mapping_RDFGraph.txt', header = None, sep = '\t')
data = data.values
mapping_dict = dict(zip(data[:,1],data[:,0]))

for item in edgelist_data:
	it1 = mapping_dict[item[0]].split('/')[-1]
	it2 = mapping_dict[item[1]].split('/')[-1]
	relations.append((it1,it2))

model = PoincareModel(relations)
model.train(epochs=50,batch_size=200) # according to paper, good representations could be learned after only 20 epochs

model.kv.save_word2vec_format(data_dir+'poincare/embeddings_has_indication_poincare_minibatch_200.txt')
pdb.set_trace()
