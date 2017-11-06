import nltk
import nltk.classify.util
import nltk.classify

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from nltk.classify import DecisionTreeClassifier
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import codecs
import sys
import string

TRAIN_FILENAME = "_train.txt"
TEST_FILENAME = "_test.txt"
PREDICT_FILENAME = "_predicted.txt"

TRAIN_SET_LEN_RATIO = 0.75

def create_word_features(words):
    #useful_words = [word for word in words if (word not in stopwords.words("english") and word not in string.punctuation)]
    useful_words = [word for word in words if word not in stopwords.words("english")]
    word_dict = dict([(word, True) for word in useful_words])
    return word_dict


def transform_input_file(filename):
    result = []
    with codecs.open(filename,'r',encoding='utf8', errors="ignore") as f:
        for line in f:
            temp_line = line.strip("\r")
            splitted_line = temp_line.split("\t")
            words = word_tokenize(splitted_line[0])
            result.append((create_word_features(words), splitted_line[1]))

    train_len = int(round(len(result) * TRAIN_SET_LEN_RATIO))
    return result[:train_len],result[train_len:]

def train_and_test(train_set, test_set):
    #classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
    #classifier = nltk.classify.SklearnClassifier(SVC()).train(train_set)
    #classifier = DecisionTreeClassifier.train(train_set)
    classifier = NaiveBayesClassifier.train(train_set)

    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(accuracy * 100)

    classifier = NaiveBayesClassifier.train(train_set + test_set)
    return classifier

def predict(classifier, test_filename, predict_filename):
    with codecs.open(test_filename,'r',encoding='utf8', errors="ignore") as test_file:
        with codecs.open(predict_filename,'w',encoding='utf8', errors="ignore") as predict_file:
            for line in test_file:
                temp_line = line.strip("\r")
                words = word_tokenize(temp_line)
                words = create_word_features(words)
                result = classifier.classify(words)

                predict_file.write(temp_line + "\t" + result + "\n")


def main(args):
    train_filename = args[1] + TRAIN_FILENAME
    test_filename = args[1] + TEST_FILENAME
    predict_filename = args[1] + PREDICT_FILENAME

    train_set, train_test_set = transform_input_file(train_filename)
    classifier = train_and_test(train_set, train_test_set)
    predict(classifier, test_filename, predict_filename)


def init():
    nltk.download()

if __name__ == '__main__':
    #init()
    main(sys.argv)