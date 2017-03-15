import re
import urllib
from sklearn.externals import joblib
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
import BeautifulSoup
import wikipedia



def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    return True


def parse_text(address):
    html = urllib.urlopen(address).read()
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(visible, texts)
    txt = ""
    for i, l in enumerate(visible_texts):
        txt = txt + " " + visible_texts[i].replace('\n', '')
    return txt


LoadWikiPage = False
LoadLandingPage = False

Model_Vocabulary_File = '/home/mohammad/Desktop/Shiftkey/Model_Vocabulary_File.sav'
Model_File = '/home/mohammad/Desktop/Shiftkey/Model_File.sav'

vocabulary = joblib.load(Model_Vocabulary_File)
clf = joblib.load(Model_File)

Descriptions = open("/home/mohammad/Desktop/Shiftkey/testdescriptions", 'r')
Titles = open("/home/mohammad/Desktop/Shiftkey/testtitles", 'r')
Titles = Titles.readlines()
Addresses = open("/home/mohammad/Desktop/Shiftkey/testaddresses", 'r')
Addresses = Addresses.readlines()

from bs4 import BeautifulSoup

corpus = []
for index, line in enumerate(Descriptions):
    title = Titles[index].replace('u\'', '').replace('\'', '')

    newcontent = title + line
    if (LoadWikiPage):
        try:
            wiki = wikipedia.page(title)
            newcontent += " " + wiki.content
            break
        except:
            print "wikierr:" + title

    if (LoadLandingPage):
        address = "http://www." + Addresses[index].replace('u\'', '').replace('\'', '')
        try:
            htmltxt = parse_text(address)
            newcontent += " " + htmltxt
            break
        except:
            print "land err:" + address

    corpus.append(re.sub(r'\d+', '', newcontent))

vectorizer = CountVectorizer(min_df=2, stop_words=stop_words.ENGLISH_STOP_WORDS, lowercase=True, ngram_range=(1, 1),
                             vocabulary=vocabulary)
X = vectorizer.fit_transform(corpus)
y = clf.predict(X)

print "========\nResults:\n========\n"
for index, item in enumerate(y):
    print Titles[index].replace('u\'', '').replace('\'', ''), " ", y[index]
