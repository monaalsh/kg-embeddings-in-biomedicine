
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score

import keras
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import pandas as pd
import itertools
import pdb
import json

import tensorflow as tf
import random


np.random.seed(33)
random.seed(12345)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=""   #use CPU

from tensorflow.python.client import device_lib
print device_lib.list_local_devices()



def negcum(rank_vec):
	rank_vec_cum = []
	prev = 0
	for x in rank_vec:
		if x == 0:
			x = x+1
			prev = prev + x
			rank_vec_cum.append(prev)
		else:
			rank_vec_cum.append(prev)
	rank_vec_cum = np.array(rank_vec_cum)
	return rank_vec_cum


data_dir = '../../Documents/kg_embeddings_data/data/'

# use skiprows with walking RDFOWL embeddings and poincare embeddings only
data = pd.read_csv(data_dir+'transE/embeddings_has_indication_transE.txt', header = None, sep = ' ')
embds_data = data.values
embds_dict = dict(zip(embds_data[:,0],embds_data[:,1:]))

data = pd.read_csv(data_dir+'mapping_RDFGraph.txt', header = None, sep = '\t')
data = data.values
mapping_dict = dict(zip(data[:,1],data[:,0]))


ent1_embeddings = {}
ent2_embeddings = {}

for item in embds_dict:
	ent = item.split('/')[-1]
	if ent.startswith('CID'):
		ent1_embeddings[ent] = np.array(embds_dict[item], dtype = 'float32')
	if ent.startswith('DOID_'):
		ent2_embeddings[ent] = np.array(embds_dict[item], dtype = 'float32')
	else:
		continue

# pdb.set_trace()

train_graph = list(open(data_dir+'train_has_indication.txt').readlines())
train_edges = [item.strip().split() for item in train_graph]
train_nodes = [item[0] for item in train_edges]
train_nodes = [mapping_dict[int(it)].split('/')[-1] for it in train_nodes]
train_edges = [(mapping_dict[int(item[0])].split('/')[-1],mapping_dict[int(item[1])].split('/')[-1]) for item in train_edges]


test_graph = list(open(data_dir+'test_has_indication.txt').readlines())
test_edges = [item.strip().split() for item in test_graph]
test_edges = [(mapping_dict[int(item[0])].split('/')[-1],mapping_dict[int(item[1])].split('/')[-1]) for item in test_edges]

positive_pairs = train_edges + test_edges

associations = {}
allents1 = set()
allents2 = set()

for item in positive_pairs:
	ent1 = item[0]
	ent2 = item[1]
	if ent1 in associations:
		associations[ent1].append(ent2)
	else:
		associations[ent1] = [ent2]


ent1_ent2_tr = {}
ent1_ent2_ts = {}

for (ent1,ent2) in train_edges:
	if ent1 in ent1_embeddings and ent2 in ent2_embeddings:
		ent1_embds = ent1_embeddings[ent1]
		ent2_embds = ent2_embeddings[ent2]
		allents1.add(ent1)
		allents2.add(ent2)
		ent1_ent2_tr[(ent1,ent2)] = np.concatenate((ent1_embds,ent2_embds), axis=0)

for (ent1,ent2) in test_edges:
	if ent1 in ent1_embeddings and ent2 in ent2_embeddings:
		ent1_embds = ent1_embeddings[ent1]
		ent2_embds = ent2_embeddings[ent2]
		allents1.add(ent1)
		allents2.add(ent2)
		ent1_ent2_ts[(ent1,ent2)] = np.concatenate((ent1_embds,ent2_embds), axis=0)


allpairs = list(itertools.product(set(allents1),set(allents2)))
positive_pairs_tr = ent1_ent2_tr.keys()
positive_pairs_ts = ent1_ent2_ts.keys()


negative_pairs = set(allpairs) - set(positive_pairs_tr)
negative_pairs = set(negative_pairs) - set(positive_pairs_ts)

negative_pairs_tr = random.sample(negative_pairs, len(positive_pairs_tr))
negative_pairs = set(negative_pairs) - set(negative_pairs_tr)
negative_pairs_ts = random.sample(negative_pairs, len(positive_pairs_ts))


non_ent1_ent2_tr = {}
non_ent1_ent2_ts = {}

for (ent1,ent2) in negative_pairs_tr:
	ent1_embds = ent1_embeddings[ent1]
	ent2_embds = ent2_embeddings[ent2]
	non_ent1_ent2_tr[(ent1,ent2)] = np.concatenate((ent1_embds,ent2_embds), axis=0)


for (ent1,ent2) in negative_pairs_ts:
	ent1_embds = ent1_embeddings[ent1]
	ent2_embds = ent2_embeddings[ent2]
	non_ent1_ent2_ts[(ent1,ent2)] = np.concatenate((ent1_embds,ent2_embds), axis=0)


pos_tr = np.array(ent1_ent2_tr.values())
neg_tr = np.array(non_ent1_ent2_tr.values())

pos_ts = np.array(ent1_ent2_ts.values())
neg_ts = np.array(non_ent1_ent2_ts.values())

train_data = np.concatenate((pos_tr, neg_tr))
train_labels = np.concatenate((np.ones(len(ent1_ent2_tr), dtype = 'int32'),np.zeros(len(non_ent1_ent2_tr), dtype='int32')), axis=0)
tr_labels = keras.utils.to_categorical(train_labels, num_classes=None)

test_data = np.concatenate((pos_ts, neg_ts))
test_labels = np.concatenate((np.ones(len(ent1_ent2_ts), dtype = 'int32'),np.zeros(len(non_ent1_ent2_ts), dtype='int32')), axis=0)
ts_labels = keras.utils.to_categorical(test_labels, num_classes=None)


tf.set_random_seed(33)
model = Sequential()
model.add(Dense(512, input_dim=256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
model.fit(train_data, tr_labels,validation_split = 0.1,epochs=100,callbacks=[earlystopper], verbose=1)

pred = model.predict_proba(test_data)[:,1]

print('has_indication_transE')
print('Binary AUC: {}'.format(roc_auc_score(test_labels, pred)))


pred_labels = model.predict(test_data)
pred_labels = np.argmax(pred_labels,axis=1)
f1score = f1_score(test_labels, pred_labels)
print('f1score: {}'.format(f1score))
print('the number of train data: {}'.format(len(train_data)))
print('the number of test data: {}'.format(len(test_data)))

label_mat = {}
recall_10 = {}
ranked_entities = {}
allents = ent2_embeddings.keys()
allents2 = list(allents2) #only interacting enttities

for item in test_edges:
	ent1 = item[0]
	if ent1 in label_mat:
		continue

	ent2 = item[1]
	ents2 = associations[ent1]
	if ent1 in train_nodes:
		train_ents = [item[1] for item in train_edges if item[0]== ent1]
		test_ents = list(set(ents2) - set(train_ents))
		to_test = list(set(allents) - set(train_ents))

	else:
		test_ents = list(set(ents2))
		to_test = list(set(allents))

	if ent1 in ent1_embeddings and ent2 in ent2_embeddings:
		ents_embed = list()
		ents_embed.append(ent1_embeddings[ent1])
		ent1embds = np.array(ents_embed, dtype='float32')

		test_emds = []		
		for it in to_test:
			ent2embds = ent2_embeddings[it]
			pair_embds = np.concatenate((ent1embds[0], ent2embds), axis=0)
			test_emds.append(pair_embds)

		test_emds = np.array(test_emds, dtype='float32')
		y_pred = model.predict_proba(test_emds)[:,1]
		sorted_idx = np.argsort(y_pred)[::-1]
		sort_entities = [to_test[arg] for arg in sorted_idx]


		label_vec = [0]*len(sort_entities)
		test_ranks = []
		ranked_entities = []

		for ind in test_ents:
			if ind in sort_entities:
				idx = sort_entities.index(ind)
				label_vec[idx] = 1
				test_ranks.append(idx)
				ranked_entities.append(ind)

		label_mat[ent1] = label_vec

		if len(test_ranks) > 0:
			test_r = np.array(test_ranks,dtype='int32')
			ranked_entities = np.array(ranked_entities)
			recall_10[ent1] = len(np.where(test_r <= 10)[0])/float(len(test_r))



#get max label vec dimension to compare with
len_vec = []
for item in label_mat:
	len_vec.append(len(label_mat[item]))


col_num = max(len_vec)
array_tp = np.zeros((len(label_mat), col_num),dtype='float32')
array_fp = np.zeros((len(label_mat), col_num), dtype = 'float32')

for i,row in enumerate(label_mat.values()):
		elem = np.asarray(row, dtype='float32')
		tofill = col_num - len(row)
		tpcum = np.cumsum(elem)
		tpcum = np.append(tpcum,np.ones(tofill)*tpcum[-1])
		fpcum = negcum(elem)
		fpcum = np.append(fpcum,np.ones(tofill)*fpcum[-1])
		array_tp[i] = tpcum
		array_fp[i] = fpcum

tpsum = np.sum(array_tp, axis = 0)
fpsum = np.sum(array_fp, axis = 0)
tpr_r = tpsum/max(tpsum)
fpr_r = fpsum/max(fpsum)

file1.write('Number of test ents1: {}\n'.format(len(label_mat)))
file1.write('Number of all ents2: {}\n'.format(len(allents)))
file1.write('Ranked AUC:  {}\n'.format(auc(fpr_r, tpr_r)))


ranks = []
top10 = []

#report mean rank and top @10 as in kg methods evaluation
for item in test_edges:
	ent1 = item[0]
	ent2 = item[1]

	if ent1 in ent1_embeddings and ent2 in ent2_embeddings:
		ents_embed = list()
		ents_embed.append(ent1_embeddings[ent1])
		ent1embds = np.array(ents_embed, dtype='float32')

		if ent1 in train_nodes:
			train_ents = [item[1] for item in train_edges if item[0]== ent1]
			to_test = list(set(allents) - set(train_ents))
		else:
			to_test = list(set(allents))

		test_emds = []
		for it in to_test:
			ent2embds = ent2_embeddings[it]
			pair_embds = np.concatenate((ent1embds[0], ent2embds), axis=0)
			test_emds.append(pair_embds)

		# pdb.set_trace()
		test_emds = np.array(test_emds, dtype='float32')
		y_pred = model.predict_proba(test_emds)[:,1]
		sorted_idx = np.argsort(y_pred)[::-1]
		sort_entities = [to_test[arg] for arg in sorted_idx]
		idx = sort_entities.index(ent2)
		ranks.append(idx)
		if idx <= 10:
			top10.append(1)

print('mean ranks: {}'.format(np.mean(ranks)))
topat10 = float(np.sum(top10)/float(len(ranks)))*100
print('%the top @10: {}'.format(topat10))

#pdb.set_trace()
