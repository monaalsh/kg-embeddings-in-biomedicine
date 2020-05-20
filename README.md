# kg-embeddings-in-biomedicine

## Background
Bio-ontologies and Linked Data have become integral part of biological
and biomedical databases to assist in standardization, representation
and dissemination of knowledge in biomedicine and life sciences for
almost the past two decades. They have traditionally been utilized for
ensuring data organization, maintaining data integrity and empoIring
search capabilities in the biomedical and biological domains. This rich and versatile form of data and knowledge can be made available as **Knowledge graphs**; a multi-relational
heterogeneous graph. Recently, knowledge graphs embeddings methods
have emerged as an effective paradigm for data analytics, and
therefore, offering new potentials for building machine learning
models for prediction and decision support in health care and
medicine.
We presents a comparative assessment and a standard benchmark for knowledge graphs representation learning methods on the the task of link prediction for biological relations. We systematically investigate and compare betIen state-of-the-arts embedding methods based on carefully designed settings for training and evaluation. We test various strategies aimed at controlling the amount of information related to each relation in the knowledge graph and its effects on the final performance. The quality of the knowledge graph features are further assessed through clustering and visualization. Additionally, we employ several evaluation metrics and examine their uses and differences. Finally, we discuss the limitations and suggest some guidelines for the development and use of knowledge graphs representation learning in the bio-medical and biological domains.

## Requirements
*python >= 3.4
*keras >= 2.2
*tensorflow 1.11
*pandas 
*json
 

## Methods
The following knowledge graph embeddings methods have been evaluated in 
this work
*[TransE] (https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf): One of the Ill-known and state-of-the-art knowledge graph embeddings methods, it represents two entities and relations as vectors and measures the distance between the two entities after the relation vector transilation is applied. Efficient and fast implementation of TransE and its extensions are available at [OpenKE] (https://github.com/thunlp/OpenKE) and [KB2E] (https://github.com/thunlp/KB2E).

*[Poincare embeddings] (https://arxiv.org/pdf/1705.08039.pdf): A graph embeddings method, specifically designed to account for taxonomies and hierarchical structure, which commonly available in knowledge graphs. [Gensim] (https://radimrehurek.com/gensim/models/poincare.html) provides an efficient implementation.  

*[Walking RDF & OWL] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860058/): Knowledge graph embedding method which combines bio-ontologies and Linked Data and generates features representations for prediction and analysis and prediction in the biomedicine domain. It is available [here] (https://github.com/bio-ontology-research-group/walking-rdf-and-owl).

## Experimental Settings:

For training our models, the positive set consists of the true associations for each relation, while the negative set is constructed by sampling an equal number of negative associations from the pool of unknown associations. We strictly require the negative associations to be between entities of the same types, while the negative set is between the set of genes and diseases that are not associated. To maintain fair comparisons, I fixed the training and testing triples in all of our experiments across different methods. For each tested entity, I applied the model by fixing the first part, which corresponds to the subject and enumerating all of the objects of the same entity type. I sorted the models’ scores in descending order to obtain the rank of the correct object and reported the mean of all ranks in the test triples. We trained each method as feature generation and endto-end models:

**Features generators models**: in this mode, we employ a two-stage pipeline, which consists of treating knowledge
graphs as feature generators folloId by a link prediction
model). The aim is to assess how well the generated features
predict biological relations. Briefly, we selected neural
networks as the link prediction model, given its capability
to learn non-linear functions and reveal intricate graph patterns
encoded in pairs of feature vectors. we then used
the neural model scores produced by the sigmoid function
in the last layer to predict the object entity in the test triple.
**End-to-end models**: this corresponds to the native mode in
which knowledge graph embedding methods are trained.
I trained each method and applied the model on the test
triples. Briefly, I use the learned feature vectors or matrices
(i.e., in case of RESCAL) to compute the scoring functions.
For example, in TransE, I compute the $L1-norm$ defined
below and compare the scores of each subject in the test
triples to all of the objects (which excludes the object entities
in the training set):

$$
score(s,o) = d(s+p,o)
$$

For RESCAL, I computed the scores as defined by the loss
function as follows:

$$
score(s,o) = s M_{p}o^\top
$$

While in the Poincare method, they defined the distance in the hyperbolic space as follows:

$$
score(s,o) = arcosh\Big(1 + 2\frac{||s - o||^2}{(1-||s||)^2(1-||o||)^2}\Big)
$$

Where $arcosh$ is the inverse hyperbolic cosine and $||.||$ is the $L2-norm$.

I also design our experiments to determine how partial and free settings could affect the results of the study.

**Partial:** when using this setting, we removed 20\% of the relations links for each biological relation (i.e. for testing) but retained 80\% of these links to generate the features for each method (i.e., for training). Thereafter, we trained the neural network model on each pair of entities in the relation or applied the knowledge graph embedding method in the native mode and computed the scoring functions in the end-to-end models. Learning from known associations when predicting future or possible links resembles a widely used principle (known as guilt-by-association) for evaluating computational predictions in biomedical applications.
**Free:** when using this setting, all relations links were removed to generate the features for the nodes in the relations. As this knowledge graph is heterogeneous and multi-relational, the nodes retain connections via other relations. In contrast to the previous simplified setting, this resembles a more realistic but challenging approach and, therefore, could provide a more reliable and robust evaluation scheme for practical application.

## Knowledge graph

The knowledge graph is built using RDF graphs from combining several databases as shown in the table below along with the following bio-ontoogies:

* Disease ontology [(DO)](https://disease-ontology.org/downloads/)
* Gene function [(GO)](http://geneontology.org/docs/download-ontology/)
* Human phenotype ontology [(HPO)](https://hpo.jax.org/app/download/ontology)



| Relation 	   | Relation database source  | Source type | Target type |
 | --------------- | ----------- |:---------------:|:--------------:|
 | has function  |  [EBI Gene Ontology Annotation Database] (http://current.geneontology.org/products/pages/downloads.html) | Gene (Entrez) | Function (Gene Ontology) |
 | has disease annotation | [DisGeNet] (https://www.disgenet.org/downloads) | Gene (Entrez) | Disease (Disease Ontology) |
 | has interaction  | [STRING] (https://string-db.org/cgi/download.pl) | Gene (Entrez) | Gene (Entrez) |
 | has sideeffect  | [SIDER] (http://sideeffects.embl.de/download/) | Drug (PubChem) | Phenotype (Human phenotype)|
 | has indication | [SIDER] (http://sideeffects.embl.de/download/) | Drug (PubChem) | Disease (Disease Ontology) | 
 | has target |  [STITCH] (http://stitch.embl.de/) | Gene (Entrez) | Drug (PubChem) |
 | has gene phenotype | [HPO annotations] (https://hpo.jax.org/app/download/annotation) | Gene (Entrez) | Phenotype (Human Phenotype Ontology) |
| has disease phenotype | [Hoehndorf R. et al] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4458913/?report=reader) | Disease (Disease ontology) | Phenotype (Human Phenotype Ontology)| 

The data was normalized and we mapped all database identifiers to their ontology identifiers as described in [Alshahrani et al] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860058/)


## Data and Embeddings
All the knowledge graph data and embeddings related to this study are available [here](http://bio2vec.net/data/kg_embeddings/)
