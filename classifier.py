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


readDate = open("/home/mohammad/Desktop/test", 'r')
readDateTitles = open("/home/mohammad/Desktop/titles", 'r')
readDateTitles =readDateTitles.readlines()
readDateAddr = open("/home/mohammad/Desktop/addr", 'r')
readDateAddr =readDateAddr.readlines()



corpus = []
for index, line in enumerate(readDate):

    title = readDateTitles[index].replace('u\'', '').replace('\'', '')
    address = "http://www."+readDateAddr[index].replace('u\'', '').replace('\'', '')
    #html = html2text.html2text(urllib2.urlopen(address).read())
    try:

        #html=html2text.html2text(urllib2.urlopen(address).read())
        print "ok: "+title
        ny = wikipedia.page(title)
        newcontent=line+" "+ny.content
        #newcontent = line #+ " " + html


    except :
        print "exception :"+title
        newcontent=line

    corpus.append(re.sub(r'\d+', '',newcontent))

vectorizer = CountVectorizer(min_df=2, stop_words=stop_words.ENGLISH_STOP_WORDS, lowercase=True, ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)

# print vectorizer.get_feature_names()

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)

Y = []
readlabels = open("/home/mohammad/Desktop/labels", 'r')
for index, line in enumerate(readlabels):
    Y.append(line.replace('\n', '').replace('\'',''))

# clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
clf = MLPClassifier(verbose=0, random_state=0)
# clf = GaussianNB()
# clf = KNeighborsClassifier(3)
# clf = GaussianProcessClassifier(warm_start=True)
# clf.fit(X, Y)
# print clf.score(X, Y)

scores = cross_val_score(clf, X, Y, cv=4)
print "Accuracy: " , scores.mean() , " +/-" , scores.std() * 2

# print clf
# clf.predict([[2., 2.]])







