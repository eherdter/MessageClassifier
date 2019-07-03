import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath):

    ''' Loads data from database.'''
    ''' Returns: X, Y, and category names. '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    X= df.message.values
    Y = df.iloc[:, :35].values
    categories = df.iloc[:, :35].columns

    return X, Y, categories


def tokenize(text):

    ''' Tokenizer function that processes the message data.'''
    ''' Returns: the cleaned tokens for each message. '''

    #replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #Normalize (lowercase and remove punctuation)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #Tokenize text
    tokens = word_tokenize(text)

    #Lemmatize and remove stopwords, end with stemming
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens if tok not in stopwords.words("english")]
    clean_tokens = [stemmer.stem(tok) for tok in clean_tokens]

    return clean_tokens


def build_model():

    ''' Builds the ML pipeline with GridSearchCV.'''
    ''' Returns: Instantiated model that should be fit in a following function. '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(random_state=42))
    ])


    parameters = {
        #'tfidf__use_idf': (True, False),
        'clf__n_estimators':[50] #range(50,100,10),
        #'clf__min_samples_split':range(5,25,5)
    }

    model = GridSearchCV(pipeline, param_grid = parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    '''Makes a prediction and evaluates the models predictive abilities using a
        classification_report scheme.'''

    '''Prints:classification report where reported averages are a
        prevalence-weighted macro-average across classes, and precision metrics.'''

    Y_pred = model.predict(X_test)

    #loops through each category and prints classification report for each.
    for i in range(len(Y_pred.T)):
        cat = category_names[i]
        pred_cat = Y_pred.T[i]
        test_cat = Y_test.T[i]
        print(cat, classification_report(test_cat, pred_cat), precision_score(test_cat, pred_cat))

    return None



def save_model(model, model_filepath):

    ''' Saves model to pickle file.'''

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        print('Splitting data into testing and training sets using test_size = 0.2')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        print('Finished  building model.')

        print('Training model.It may take a while, please be patient.')
        model.fit(X_train, Y_train)
        print('Finished training model.')

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Finished evaluating model.')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
