
import pdb
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import random
import pandas as pd
import json

np.random.seed(33)


data_dir = 'data/'
mapping = pd.read_csv(data_dir+'mapping_RDFGraph.txt', header = None, sep = '\t')
mapping = mapping.values
mapping_dict = dict(zip(mapping[:,1],mapping[:,0]))


graph = [line.strip().split() for line in open(data_dir+"edgelist_RDFGraph.txt")]
sub_links = [(item[0],item[1],item[2]) for item in graph if 'has_function' in mapping_dict[int(item[2])]]
graph = [(item[0],item[1],item[2]) for item in graph]
graphorig = set(graph) 

print('All graph edges before dropping: {}'.format(len(graph)))
print('subgraph edges: {}'.format(len(sub_links)))
data = np.array(sub_links)

# pdb.set_trace()
train_edges, test_edges = train_test_split(data, test_size=0.2, random_state=42)

test = [(item[0],item[1],item[2]) for item in test_edges]
rdfgraph = set(graph) - set(test)


edgelist = data_dir+ 'edgelist_has_function.txt'
train_file = data_dir+ 'train_has_function.txt'
test_file = data_dir+ 'test_has_function.txt'
newrdf = list(rdfgraph)
newrdf = np.array(newrdf)
np.savetxt(edgelist, newrdf, fmt = '%s')
np.savetxt(train_file, train_edges, fmt = '%s')
np.savetxt(test_file, test_edges, fmt = '%s')

print('all graph edges after dropping 20 percent: {}'.format(len(rdfgraph)))
print('train graph edges is: {}'.format(len(train_edges)))
print('test graph edges is: {}'.format(len(test_edges)))	

pdb.set_trace()
