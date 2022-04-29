# Ulya Bayram
# this code is to apply another pass through the words in the file, some punctuations aren't properly removed during the first pass
# there are words like 'i, 'yes, 'then, 'you, etc. remaining, also there are *e, *f, for some reason
# these two cases will be considered and removed here.
import glob
import sys
import codecs
import re
import os

def removeSpaceFromWords(tlist):

    for wordi in range(len(tlist)):
        word = tlist[wordi]
        if ' ' in tlist[wordi]:
            word = word.replace(' ', '')

        tlist[wordi] = word
    return tlist

def cleanStart(filelist):
    newlist = []

    for file_ in filelist:
        newlist.append(file_.split('/')[1])

    return newlist

###################################################################################
# save all chapters, all pre-processed but found plenty of errors, here pre-processing over the faults

#reload(sys)
#sys.setdefaultencoding('utf-8')

fo_train = open('train_1.txt', 'r')
train_filenames = fo_train.read().split('\n')
fo_train.close()
fo_test = open('test_1.txt', 'r')
test_filenames = fo_test.read().split('\n')
fo_test.close()

selected_filenames = train_filenames + test_filenames
# check whether training and test filenames overlap
print(len(selected_filenames))
print( len( list(set(selected_filenames)) ) )

glob_sentence = glob.glob('House114_sentences/*.txt')
glob_unigrams = glob.glob('House114_unigrams/*.txt')

# clean up the sentence files
for file_ in glob_sentence:

    if file_.split('/')[1] in selected_filenames: # qualifies for cleaning up
        f_file = open(file_, 'r')
        txt_ = f_file.read().split(' ')
        f_file.close()

        # clean the numericals from this text - convert them to 17 - my constant
        for word_i in range(len(txt_)):
            if txt_[word_i].isdigit():
                # convert all digits within text to 17
                txt_[word_i] = '17'
        f_file = open(file_, 'w')
        f_file.write(' '.join(txt_))
        f_file.close()
        #print(' '.join(txt_))
    else: # delete this file - unused
        os.system('rm ' + file_)

# clean up the unigram files
for file_ in glob_unigrams:

    if file_.split('/')[1] in selected_filenames: # qualifies for cleaning up
        f_file = open(file_, 'r')
        txt_ = f_file.read().split(' ')
        f_file.close()

        # clean the numericals from this text - convert them to 17 - my constant
        for word_i in range(len(txt_)):
            if txt_[word_i].isdigit():
                # convert all digits within text to 17
                txt_[word_i] = '17'
        f_file = open(file_, 'w')
        f_file.write(' '.join(txt_))
        f_file.close()
    else: # delete this file - unused
        os.system('rm ' + file_)
