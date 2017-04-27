
#For Preprocessing
import nltk
import pandas as pd
import string
from nltk.corpus import stopwords

#For Vectorizing and Weighting
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#For Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

#For Streamlining
from sklearn.pipeline import Pipeline

#For Predictions
from sklearn.metrics import classification_report


def text_process(s):
    nopunc =  s.translate(None, string.punctuation)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') ]

def main():
    messages = pd.read_csv('SMSSpamCollection', sep = '\t', names = ["label", "message"])

    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.1)

    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer = text_process)), 
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()), 
    ])

    pipeline.fit(msg_train, label_train)

    predictions = pipeline.predict(msg_test)
    print classification_report(predictions, label_test)

    #should be around 97% accuracy

if __name__ == '__main__':
    main()
    
    








