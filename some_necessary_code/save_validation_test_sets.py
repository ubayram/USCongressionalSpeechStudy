# Ulya Bayram

import glob
import codecs
import sys
import random
import math

def cleanLocationInformation(filename_list):

    new_list = []
    for filename_ in filename_list:
        new_list.append(filename_.split('/')[1])

    return new_list

def separateDataByLabels(fileset):

    d_files = []
    r_files = []

    for file_ in fileset:
        if 'd_' in file_[:2]:
            d_files.append(file_)
        elif 'r_' in file_[:2]:
            r_files.append(file_)

    return d_files, r_files

# save more test-training splits all randomly performed/selected
# no rules

list_of_stuff = ['97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114']

for i_death in list_of_stuff:
    fo = open('test'+i_death+'_1.txt', 'r')
    complete_test_set = fo.read().split('\n')
    d_files, r_files = separateDataByLabels(complete_test_set)

    print(d_files)
    print('\n')
    print(r_files)

    # loop here, number of splits to perform
    num_test = len(d_files)/2

    # every time you loop, shuffle all the filenames again
    random.shuffle(d_files)
    random.shuffle(r_files)

    # make selections, 75 random test samples, rest is to be used as training
    selected_test_files_a = random.sample(d_files, num_test)
    selected_test_files_c = random.sample(r_files, num_test)

    remaining_a = list(set(d_files) - set(selected_test_files_a))
    remaining_c = list(set(r_files) - set(selected_test_files_c))

    fo_val = open('validation'+i_death+'.txt', 'w')
    fo_test = open('test'+i_death+'.txt', 'w')

    # loop over all the filenames in the corpus, use if statements to find which are test
    c = 0
    # write the suicidal files first
    for file_ in d_files:

        if file_ in selected_test_files_a:
            fo_test.write(file_ + '\n')
        if file_ in remaining_a:
            fo_val.write(file_ + '\n')

    # write the control files next
    for file_ in r_files:

        if file_ in selected_test_files_c:
            fo_test.write(file_ + '\n')
        if file_ in remaining_c:
            fo_val.write(file_ + '\n')

    fo_val.close()
    fo_test.close()
