# Ulya Bayram
# Classify unigram features
# ulyabayram@gmail.com
import math
import numpy as np
#import time
import classifier_functions_nn as cf
import scipy.stats
from scipy import stats

###########################################################################

# Read the current classifier's save directory
read_dir = cf.getReadDirectory()
save_feature_dir = cf.getFeatureSaveDirectory()
feature_type = 'unigram'

for i_split in ['97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114']:

    print('Split = ' + str(i_split))

    # read the features corresponding to the columns in these feature matrices
    fo_columns = open('../features'+i_split+'/'+feature_type+'_matrix_columnames.txt', 'r')
    feature_names = fo_columns.read().split('\n')[:-1]
    fo_columns.close()

    # read the previously trained and pickled classifier
    model = cf.readTheClassifier(read_dir, feature_type, i_split)

    # read the feature importances from the trained model, and save them
    savefilename_ = save_feature_dir + feature_type + '_feature_importance_h'+i_split+'.txt'

    cf.saveFeatureImportances(savefilename_, model, feature_names)
