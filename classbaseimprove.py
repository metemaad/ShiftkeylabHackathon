from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import wikipedia
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
import html2text

import urllib2
vocabtest=	["business","buy","case study","client","company","consult","consultation","coupon","customer","development","enterprise","family","growth","professional","product","redeem",	"sign up","service","shop","solution","support","trial","trust","you","your"]


readData = open("/home/mohammad/Desktop/test", 'r')
readDataTitles = open("/home/mohammad/Desktop/titles", 'r')
readDataTitles =readDataTitles.readlines()

readDatagrp = open("/home/mohammad/Desktop/grp", 'r')
readDatagrp =readDatagrp.readlines()

readDataaddr = open("/home/mohammad/Desktop/addr", 'r')
readDataaddr =readDataaddr.readlines()




corpus = []
for index, line in enumerate(readData):
    title = readDataTitles[index].replace('u\'', '').replace('\'', '')
  #  print index
    grp= readDatagrp[index].replace('u\'', '').replace('\'', '')
    #addr = readDataaddr[index].replace('u\'', '').replace('\'', '')
    newcontent=title+grp+line
#    print len(line),len(title)
    corpus.append(re.sub(r'\d+', '',newcontent))

vectorizer = CountVectorizer(min_df=2, stop_words=stop_words.ENGLISH_STOP_WORDS, lowercase=True, ngram_range=(1, 1),vocabulary=vocabtest)
X = vectorizer.fit_transform(corpus)
#print X

vocabulary= vectorizer.get_feature_names()

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)

Y = []
readlabels = open("/home/mohammad/Desktop/labels", 'r')
for index, line in enumerate(readlabels):
    Y.append(line.replace('\n', '').replace('\'',''))



clf = MLPClassifier(verbose=0, random_state=0)
scores = cross_val_score(clf, X, Y, cv=4)
print "MLPClassifier:",scores.mean()
print "MLPClassifier:",
clf = MLPClassifier(verbose=0, random_state=0).fit(X,Y)

#clf = svm.SVC(kernel='linear')
#scores = cross_val_score(clf, X, Y, cv=4)
#print "SVM linear Classifier:",scores.mean()

##clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
#scores = cross_val_score(clf, X, Y, cv=4)
#print "SVM RBF Classifier:",scores.mean()

#clf = svm.SVC(kernel='linear')
#scores = cross_val_score(clf, X, Y, cv=4)
#print "SVM RBF Classifier:",scores.mean()

#clf = GaussianNB()# TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
#scores = cross_val_score(clf, X, Y, cv=4)
#print "GaussianNB Classifier:",scores.mean()


#clf = KNeighborsClassifier(3)#TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
#scores = cross_val_score(clf, X, Y, cv=4)
#print "KNeighbors Classifier:",scores.mean()

#clf = GaussianProcessClassifier(warm_start=True)
#scores = cross_val_score(clf, X, Y, cv=4)
#print "GaussianProcess Classifier:",scores.mean()



#print clf



readData = open("/home/mohammad/Desktop/test", 'r')
readDataTitles = open("/home/mohammad/Desktop/titles", 'r')
readDataTitles =readDataTitles.readlines()




corpus = []
for index, line in enumerate(readData):
    title = readDataTitles[index].replace('u\'', '').replace('\'', '')
 #   print index
    grp= readDatagrp[index].replace('u\'', '').replace('\'', '')
    #addr = readDataaddr[index].replace('u\'', '').replace('\'', '')
    newcontent=title+line
 #   print len(line),len(title)
    corpus.append(re.sub(r'\d+', '',newcontent))

vectorizer = CountVectorizer(min_df=2, stop_words=stop_words.ENGLISH_STOP_WORDS, lowercase=True, ngram_range=(1, 1),vocabulary=vocabulary)
X = vectorizer.fit_transform(corpus)
#print X
y=clf.predict(X)
for index,item in enumerate(Y):
    if (y[index]!=Y[index]):
        print index








