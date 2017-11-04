import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import codecs

AMAZON_TRAIN_FILENAME = "amazon_train.txt"
AMAZON_TEST_FILENAME = "amazon_test.txt"

YELP_TRAIN_FILENAME = "yelp_train.txt"
YELP_TEST_FILENAME = "yelp_test.txt"

def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    word_dict = dict([(word, True) for word in useful_words])
    return word_dict


def transform_input_file(filename):
    result = []
    with codecs.open(filename,'r',encoding='utf8') as f:
        for line in f:
            temp_line = line.strip("\r")
            splitted_line = temp_line.split("\t")
            words = word_tokenize(splitted_line[0])
            result.append((create_word_features(words), splitted_line[1]))

    return result


def main():
    amazon_train = transform_input_file(AMAZON_TRAIN_FILENAME)

def init():
    nltk.download()

if __name__ == '__main__':
    #init()
    main()