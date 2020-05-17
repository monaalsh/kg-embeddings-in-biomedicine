import config
import models
import tensorflow as tf
import numpy as np
import pdb


con = config.Config()

data_dir = '../../Documents/kg_embeddings_data/data/rescal/has_indication/'

con.set_in_path(data_dir)
#True: Input test files from the same folder.
# con.set_test_link_prediction(True)
# con.set_test_triple_classification(True)
# pdb.set_trace()

con.set_work_threads(1)
con.set_train_times(100)
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(0)
con.set_margin(1)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adam") 
#Models will be exported via tf.Saver() automatically.
con.set_export_files(data_dir +"rescal_model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files(data_dir+"rescal_embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.RESCAL)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
#con.test()
# pdb.set_trace()
