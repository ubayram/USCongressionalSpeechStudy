# Ulya Bayram
# Classify unigram features
# ulyabayram@gmail.com
import math
import numpy as np
#import time
import classifier_functions_nn as cf
import pickle
import scipy.stats
from scipy import stats

###########################################################################

# Read the current classifier's save directory
save_dir = cf.getSaveDirectory()
feature_type = 'unigram'

for i_split in ['98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']: #, 
    # 1) create the files to write the overall classification results to
    f_split_lvl_val = open(save_dir + 'validation_overall_classified_'+feature_type+'_matrix_results_h'+ str(i_split) + '.txt', 'w')
    f_split_lvl_val.write('TP\tTN\tFP\tFN\tprecision\trecall\taccuracy\tF1\tAUC\tDeLongCI\n')

    f_split_lvl_test = open(save_dir + 'test_overall_classified_'+feature_type+'_matrix_results_h'+ str(i_split) + '.txt', 'w')
    f_split_lvl_test.write('TP\tTN\tFP\tFN\tprecision\trecall\taccuracy\tF1\tAUC\tDeLongCI\n')

    # 2) create the files to write the document level classification results of: training, validation, test
    f_results_save_train = open(save_dir + 'Training/training_classified_'+feature_type+'_matrix_results_house'+ str(i_split) + '.txt', 'w')
    f_results_save_train.write('filename\ttrueclass\tpredclass_prob\tnumfeatures\n')

    f_results_save_val = open(save_dir + 'Validation/validation_classified_'+feature_type+'_matrix_results_house'+ str(i_split) + '.txt', 'w')
    f_results_save_val.write('filename\ttrueclass\tpredclass_prob\tnumfeatures\n')

    f_results_save_test = open(save_dir + 'Test/test_classified_'+feature_type+'_matrix_results_house'+ str(i_split) + '.txt', 'w')
    f_results_save_test.write('filename\ttrueclass\tpredclass_prob\tnumfeatures\n')

    # 3) read the training, validation, and test set document lists
    f_test = open('train'+ str(i_split) + '.txt', 'r')
    all_training_files = f_test.read().split()
    f_test.close()
    f_test = open('validation'+ str(i_split) + '.txt', 'r')
    all_validation_files = f_test.read().split()
    f_test.close()
    f_test = open('test'+ str(i_split) + '.txt', 'r')
    all_test_files = f_test.read().split()
    f_test.close()

    # separate the training and validation set files into their respective classes
    training_d_files, training_r_files = cf.separateGroupFiles(all_training_files)
    validation_d_files, validation_r_files = cf.separateGroupFiles(all_validation_files)

    # read the complete feature matrices
    feature_matrix = np.loadtxt('features'+ str(i_split) + '/'+feature_type+'_feature_matrix.gz')

    # apply preprocessing to the feature matrices before we move onto other things
    feature_matrix = cf.applyFeaturePreprocessing(feature_matrix)
  
    # read the filenames corresponding to the rows of above feature matrix
    f_rownames = open('features'+ str(i_split) + '/feature_matrix_row_filenames.txt')
    filenames_rows = f_rownames.read().split('\n')[:-1]
    f_rownames.close()

    # split the feature matrix into training, validation and test, and collect their class labels
    matrices, labels, row_filenames = cf.splitFeatureMatrix(feature_matrix, filenames_rows, training_d_files, training_r_files,
                                                                validation_d_files, validation_r_files, all_test_files)
    training_matrix = matrices['train']
    validation_matrix = matrices['validation']
    test_matrix = matrices['test']

    training_labels = labels['train']
    validation_labels = labels['validation']
    test_labels = labels['test']

    training_row_filenames = row_filenames['train']
    validation_row_filenames = row_filenames['validation']
    test_matrix_row_filenames = row_filenames['test']

    # some classifiers might have some parameter selection (classifier model building) requirement
    # if so, do it here, now - and use the selected parameters
    # if classifier has no such dependence, params_ will be an empty list [] and classifier functions will ignore it
    #params_ = cf.getClassifierParams(training_matrix, validation_matrix, training_labels, validation_labels)

    # train the classifier - save the trained model - in case it might be necessary to re-use it in the future
    #model = cf.trainTheClassifier(training_matrix, training_labels, params_)
    #filename = str(save_dir + 'models/'+feature_type+'_trained_model_h'+ str(i_split) + '.sav')
    #pickle.dump(model, open(filename, 'wb'))

    model = cf.trainTheClassifier(training_matrix, training_labels, validation_matrix, validation_labels)
    model_json = model.to_json()
    with open(save_dir + 'models/'+feature_type+'_trained_model_h'+ str(i_split) + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_dir + 'models/'+feature_type+'_trained_model_h'+ str(i_split) + '.h5')

    # perform the classifications using the same trained classifier
    training_classification_results_prob = cf.classifyFeatures(training_matrix.toarray(), model)
    validation_classification_results_prob = cf.classifyFeatures(validation_matrix.toarray(), model)
    test_classification_results_prob = cf.classifyFeatures(test_matrix.toarray(), model)

    # save the file level feature vector classification results
    cf.saveFileLevelClassificationResults(training_row_filenames, training_labels, training_classification_results_prob,
                                              f_results_save_train, np.shape(training_matrix)[1])
    cf.saveFileLevelClassificationResults(validation_row_filenames,validation_labels, validation_classification_results_prob,
                                              f_results_save_val, np.shape(validation_matrix)[1])
    cf.saveFileLevelClassificationResults(test_matrix_row_filenames, test_labels, test_classification_results_prob,
                                              f_results_save_test, np.shape(test_matrix)[1])

    # save the overall, split level results for the validation, test, and shor sets
    cf.saveOverallSplitLvlClassificationResults(f_split_lvl_val, validation_classification_results_prob, validation_labels)
    cf.saveOverallSplitLvlClassificationResults(f_split_lvl_test, test_classification_results_prob, test_labels)

    f_split_lvl_val.close()
    f_split_lvl_test.close()
