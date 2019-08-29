# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import nltk, re, string
import scipy.io as sio
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def read_email(file):
    file = open(file, 'rt')
    email = file.read()
    file.close()
    return email

def getVocabList(df_path, reverse=False):
    with open(df_path) as f:
        vocab_dict = {}
        for line in f:                    
            (val, key) = line.split()     
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key
    return vocab_dict

def preprocess(email):
    email = email.lower()
    email = re.sub('<[^<>]+>', ' ', email);
    email = re.sub('[0-9]+', 'number', email)
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email); 
    email = re.sub('[$]+', 'dollar', email);
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    table  = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens] # remove punctuation from each word
    tokenList = []
    stemmer   = PorterStemmer()
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)#Remove any non alphanumeric characters
        stemmed = stemmer.stem(token) #Use the Porter stemmer to stem the word
        if not len(token): continue #Throw out empty tokens
        tokenList.append(stemmed) #Store a list of all unique stemmed words
    return tokenList

def wordToIndices(tokenList, vocab_dict):
    word_indices = []
    for token in tokenList:
        if token in vocab_dict:
            value = vocab_dict[token]
            word_indices.append(value)
    return word_indices

def emailFeatures(indexList, vocabList):
    n = len(vocabList)
    x = np.zeros((n,1))
    for idx in indexList:
        x[idx] = 1
    return x

def preprocessing_model(email,df_path_vocab):
    tokenList = preprocess(email)
    vocab_dict = getVocabList(df_path_vocab,reverse=False)
    indexList = wordToIndices(tokenList, vocab_dict)
    x = emailFeatures(indexList,vocab_dict)
    
    print("Length of feature vector is %d" % len(x))
    print("Number of non-zero entries is: %d" % sum(x==1))
    return x

def email_classifier(model,processed_email_fv):
    email_fv   = processed_email_fv.reshape(1,-1) # for a single training example
    prediction = model.predict(email_fv)
    if prediction[0] == 0:
        print("SVM has classified email as NOT A SPAM")
    else:
        print("SVM has classified email as SPAM")

def main():
    print("\nProgramming Exercise 6: Spam Classification\n")

    # Preprocess email sample 1
    input("Press <ENTER> key to read in 'emailSample1.txt' ...")
    email = read_email('data/emailSample1.txt')
    print("\n",email)
    input("Press <ENTER> key to preprocess 'emailSample1.txt' ...")
    processed_email = preprocess(email)
    print("\n",processed_email)
    input("\nPress <ENTER> key to compute word indices of processed 'emailSample1.txt' ...")
    vocab_dict = getVocabList('data/vocab.txt', reverse=False)
    word_indices = wordToIndices(processed_email, vocab_dict)
    print("\n",word_indices)

    # Extracting features of 'emailSample1.txt'
    input("\nPress <ENTER> key to extract features of 'emailSample1.txt' ...")
    features = emailFeatures(word_indices,vocab_dict)
    print('\nLength of feature vector:', len(features))
    print('Number of non-zero entries:', sum(features > 0)[0])
    
    # Training dataset
    input("\nPress <ENTER> key to load training dataset ...")
    train_data = sio.loadmat('data/spamTrain.mat')
    X = train_data['X']
    y = train_data['y'].ravel()
    print("\n... training data loaded successfully!\n")

    # Test dataset
    input("\nPress <ENTER> key to load test dataset ...")
    test_data = sio.loadmat('data/spamTest.mat')
    Xtest = test_data['Xtest']
    ytest = test_data['ytest'].ravel()
    print("\n... test data loaded successfully!\n")

    # Find indices of spam (positive) and ham (negative) examples
    input("\nPress <ENTER> key to find the number of spam and ham email ...")
    pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
    print('\nTotal number of training emails = ',X.shape[0])
    print('Number of training spam emails = ',pos.shape[0])
    print('Number of training nonspam emails = ',neg.shape[0])

    # Setting up SVM for email classification
    input("\nPress <ENTER> key to fit SVM model for email classification ...")
    svm_clf = LinearSVC(C=0.1, loss="hinge")
    model = svm_clf.fit(X, y)
    print("\n... SVM fitted successfully!\n")

    # Training Linear SVM (Spam Classification)
    input("\nPress <ENTER> key to compute training accuracy ...")
    train_predictions = model.predict(X).reshape(y.shape[0],1)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(train_predictions, y)]  
    accuracy = 100. * (sum(map(int, correct)) / len(correct))  
    print("\nTraining accuracy: %.2f"%(accuracy)+"%")

    # Evaluating the trained Linear SVM on a test set
    input("\nPress <ENTER> key to compute test accuracy ...")
    test_predictions = model.predict(Xtest).reshape(ytest.shape[0],1)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(test_predictions, ytest)]  
    accuracy = 100. * (sum(map(int, correct)) / len(correct))  
    print("\nTest accuracy: %.2f"%(accuracy)+"%")

    #Sort indices from most important to least-important (high to low weight)
    input("\nPress <ENTER> key to output the TOP 15 predictors for spam ...")
    vocab_dict_flipped = getVocabList('data/vocab.txt',reverse=True)
    sorted_indices = np.argsort(model.coef_, axis=None )[::-1]
    sorted_weights = np.sort(model.coef_, axis=None )[::-1]
    print([ vocab_dict_flipped[x] for x in sorted_indices[:10] ])

    # Classifying the emails
    input("\nPress <ENTER> key to classify 'emailSample1.txt' ...")
    email1 = 'data/emailSample1.txt'
    processed_email = preprocessing_model(email,'data/vocab.txt')
    email_classifier(model,processed_email)

    input("\nPress <ENTER> key to classify 'emailSample2.txt' ...")
    email2 = 'data/emailSample2.txt'
    processed_email = preprocessing_model(email2,'data/vocab.txt')
    email_classifier(model,processed_email)

    input("\nPress <ENTER> key to classify 'spamSample1.txt' ...")
    email3 = 'data/spamSample1.txt'
    processed_email = preprocessing_model(email3,'data/vocab.txt')
    email_classifier(model,processed_email)

    input("\nPress <ENTER> key to classify 'spamSample2.txt' ...")
    email4 = 'data/spamSample2.txt'
    processed_email = preprocessing_model(email4,'data/vocab.txt')
    email_classifier(model,processed_email)

    # Terminate program
    input("\nPress <ENTER> key to terminate program ...")

if __name__ == "__main__":
    main()
