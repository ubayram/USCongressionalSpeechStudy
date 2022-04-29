# author: Ulya Bayram
# pre-process and clean up the house 114 speech data and save the sentence-separated version
import glob
import re

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
# ------------------------------------------------------------------------------------------------------------

files_ = glob.glob('House_108/*.txt')

for file_ in files_:

    fo = open(file_, 'r')
    txt_ = fo.read()
    fo.close()

    # delete some unnecessary stuff
    txt_ = txt_.replace('Mr Speaker. ', '')
    txt_ = txt_.replace('Mr Speaker, ', '')
    txt_ = txt_.replace('Madam Speaker. ', '')
    txt_ = txt_.replace('Madam Speaker, ', '')
    txt_ = txt_.replace('Mr Chairman. ', '')
    txt_ = txt_.replace('Mr Chairman, ', '')
    txt_ = txt_.replace('Madam Chair. ', '')
    txt_ = txt_.replace('Madam Chair, ', '')
    txt_ = txt_.replace('\n.', '')
    txt_ = txt_.replace('-', " ")
    #print(txt_)
    # delete some symbols if they exist within the text
    txt_ = re.sub(r'\[(.*?)\]', '', txt_)

    txt_ = re.sub(r'\,|\(|\)|\[|\]|\*|\&|\$|\@|\%|\-|\:|\"|\_|\+|\#|\;|\\', ' ', txt_)

    txt_ = txt_.replace('..', '.') # replace multiple periods with a single one

    # replace periods with our sentence boundary signs
    txt_ = txt_.replace('.', ' * ')
    txt_ = txt_.replace('?', ' * ')
    txt_ = txt_.replace('!', ' * ')
    txt_ = re.sub(r' +', ' ', txt_)

    txt_ = txt_.lower()

    # clean the numericals from this text - convert them to 17 - my constant
    text_ = txt_.split(' ')
    for word_i in xrange(len(text_)):
        if text_[word_i].isdigit():
            # convert all digits within text to 17
            text_[word_i] = '17'

    fo = open('cleanHouse108/' + file_.split('/')[1], 'w')
    fo.write(' '.join(text_))
    fo.close()

