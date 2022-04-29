# author: Ulya Bayram
# Here are the used classification methods are defined as separate functions
# calling them simply will provide necessary information for different feature types' classification in parallel
# ulyabayram@gmail.com
import numpy as np
import math
from scipy import sparse
import keras
import compute_AUC_functions as auc
import scipy.stats
from scipy import stats
import pickle

# ---------------------------- FUNCTIONS THAT ARE CONSTANTS ACCROSS DIFFERENT CLASSIFIERS - DONT MODIFY -----------
def splitFeatureMatrix(feature_matrix, feature_filenames, training_sui_files, training_cont_files,
                           validation_sui_files, validation_cont_files, test_files):

    # initialize corresponding future feature matrices
    num_features = np.shape(feature_matrix)[1]

    # doesn't hurt to shrink the memory usage - should increase the runtime speed
    training_matrix = sparse.lil_matrix( np.zeros( (len(training_sui_files + training_cont_files), num_features) ) )
    validation_matrix = sparse.lil_matrix( np.zeros( (len(validation_sui_files + validation_cont_files), num_features) ) )
    test_matrix = sparse.lil_matrix( np.zeros( (len(test_files), num_features) ) )

    # split the filenames by their classes and initialize the class label lists
    test_sui_files, test_cont_files = separateGroupFiles(test_files)

    training_labels = []
    validation_labels = []
    test_labels = []

    training_rows = []
    validation_rows = []
    test_rows = []
    for i_file in xrange(len(feature_filenames)):

        filename = feature_filenames[i_file]
        feature_vector = feature_matrix[i_file, :]

        if filename in training_sui_files + training_cont_files:
            training_matrix[len(training_labels), :] = feature_vector
            training_rows.append(filename)

            if filename in training_sui_files:
                training_labels.append(1)
            elif filename in training_cont_files:
                training_labels.append(0)
            else:
                print('Error in training filenames')
        elif filename in validation_sui_files + validation_cont_files:
            validation_matrix[len(validation_labels), :] = feature_vector
            validation_rows.append(filename)

            if filename in validation_sui_files:
                validation_labels.append(1)
            elif filename in validation_cont_files:
                validation_labels.append(0)
            else:
                print('Error in validation filenames')
        elif filename in test_files:
            test_matrix[len(test_labels), :] = feature_vector
            test_rows.append(filename)

            if filename in test_sui_files:
                test_labels.append(1)
            elif filename in test_cont_files:
                test_labels.append(0)
            else:
                print('Error in test filenames')
        else:
            print('Error in filenames')

    matrices = {}
    matrices['train'] = training_matrix
    matrices['validation'] = validation_matrix
    matrices['test'] = test_matrix

    labels = {}
    labels['train'] = training_labels
    labels['validation'] = validation_labels
    labels['test'] = test_labels

    row_filenames = {}
    row_filenames['train'] = training_rows
    row_filenames['validation'] = validation_rows
    row_filenames['test'] = test_rows
    return matrices, labels, row_filenames

# converts the feature vector into a vector with magnitude 1
def rowNormalizeMatrix(matrix_):

    row_wise_sqrts = np.sqrt(np.sum(np.square(matrix_), axis=1))
    matrix_2 = np.copy(matrix_)

    # now divide each row items with the corresponding square root values
    for i_row in xrange(np.shape(matrix_2)[0]):
        if row_wise_sqrts[i_row] != 0:
            matrix_2[i_row, :] = matrix_2[i_row, :] / row_wise_sqrts[i_row]

    return matrix_2

# This function is constant accross all features and all classifiers
# here, all feature values are scaled, and row normalized
def applyFeaturePreprocessing(feature_matrix):

    # out default scaling method is the conversion of feature values to log domain
    # this shrinks the magnitude-related differences between feature types
    # also, keeps the feature value sign intact - which standard scaling methods fail to do so
    feature_matrix = np.sign(feature_matrix)*np.log( np.abs(feature_matrix) + 1 )
    feature_matrix = rowNormalizeMatrix(feature_matrix)

    return feature_matrix

def saveFileLevelClassificationResults(input_files, input_truelabels, input_results_prob,
                                  f_results_save_input, i_split, input_numfeatures):
    ic = 0

    if len(input_results_prob[0]) > 1: # if there are more than two values in the first item of the probs list
        input_results_prob = input_results_prob[:, 1] # take the second columns only
    else:
        input_results_prob = input_results_prob[:, 0]

    for file_ in input_files:
        # 'filename\tsplitnum\ttrueclass\tewd_result\tpredclass_prob\tnumfeatures\n'
        # 'filename\tsplitnum\ttrueclass\tpredclass_prob\tnumfeatures\n'
        f_results_save_input.write(file_ + '\t' + str(i_split) + '\t' + str(input_truelabels[ic]) +
                                       '\t' + str(input_results_prob[ic]) + '\t' + str(input_numfeatures) + '\n')
        ic += 1

    f_results_save_input.close()

def saveOverallSplitLvlClassificationResults(f_split_lvl_val, input_results_prob, input_truelabels, i_split):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    if len(input_results_prob[0]) > 1: # if there are more than two values in the first item of the probs list
        input_results_prob = input_results_prob[:, 1] # take the second columns only
    else:
        input_results_prob = input_results_prob[:, 0]

    for i_file in xrange(len(input_truelabels)):

        if input_truelabels[i_file] == 1 and input_results_prob[i_file] >= 0.5:
            TP += 1
        elif input_truelabels[i_file] == 0 and input_results_prob[i_file] < 0.5:
            TN += 1
        elif input_truelabels[i_file] == 1 and input_results_prob[i_file] < 0.5:
            FN += 1
        elif input_truelabels[i_file] == 0 and input_results_prob[i_file] >= 0.5:
            FP += 1
        else:
            print('error in computing classification scores')

    precision_ = TP / float(TP + FP)
    recall_ = TP / float(TP + FN)
    accuracy_ = (TP + TN) / float(TP + TN + FP + FN)
    f1_ = 2 * precision_ * recall_ / float(precision_ + recall_)

    # compute AUC and the 95% confidence interval
    auc_score, auc_cov_score = auc.delong_roc_variance(np.array(input_truelabels), input_results_prob.reshape(-1,))
    auc_std = np.sqrt(auc_cov_score)
    alpha = 0.95 # selected confidence level
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci_score = stats.norm.ppf(lower_upper_q, loc=auc_score, scale=auc_std)

    ci_score[ci_score > 1] = 1

    f_split_lvl_val.write(str(i_split) + '\t' + str(TP) + '\t' + str(TN) + '\t' + str(FP) + '\t' + str(FN) + '\t' + str(precision_) + '\t' +
                              str(recall_) + '\t' + str(accuracy_) + '\t' + str(f1_) + '\t' + str(auc_score) +
                              '\t' + str(ci_score[0]) + '-' + str(ci_score[1]) + '\n')

# ---------------------------- FUNCTIONS THAT ARE VARYING BY THE CHOICE OF CLASSIFIERS -----------

def getReadDirectory():
    return '../Logistic_results/'

def getFeatureSaveDirectory():
    return 'Logistic/'

# inside, read the trained model to the corresponding folder - might be needed in the future
def readTheClassifier(read_dir, feature_type, i_split):

    # load json and create model
    json_file = open(read_dir + 'models/'+feature_type+'_trained_model_h'+i_split+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(read_dir + 'models/'+feature_type+'_trained_model_h'+i_split+'.h5')

    # compile the model  using the same methods we used for training it
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
    
    return loaded_model

def saveFeatureImportances(savefilename_, trained_model, feature_names):

    # get the input-output layer weights
    layer_weights = []
    for layer in trained_model.layers:
        layer_weights.append( layer.get_weights() ) # list of numpy arrays

    #print(layer_weights[0][1][0])
    #print(layer_weights[0][1][1])
    #bias_ = layer_weights[0][1]
    btw_input_hidden_weights = layer_weights[0][0] # don't ignore the bias array

    #print(bias_)
    f_importance = open(savefilename_, 'w')
    c = 0
    for word_i in xrange(len(feature_names)):
        curr_uni_weights = btw_input_hidden_weights[c]
        f_importance.write(feature_names[word_i] + '\t' + str(curr_uni_weights[0]) + '\t' + str(curr_uni_weights[1]) +
                                     '\t' + str(curr_uni_weights[0] - curr_uni_weights[1]) + '\n')
        c += 1
    f_importance.close()

    f_importance = open(savefilename_[:-4] + '_bias.txt', 'w')
    f_importance.write(str(layer_weights[0][1][0]) + '\t' + str(layer_weights[0][1][1]))
    f_importance.close()
