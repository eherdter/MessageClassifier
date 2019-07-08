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


from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


import pickle


def load_data(database_filepath):

    ''' Loads data from database.'''
    ''' Returns: X, Y, and category names. '''


    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    
    #Select X and Y columns, as well as category names 
    X= df.message.values
    Y = df.iloc[:, :35].values
    category_names = df.iloc[:, :35].columns

    return X, Y, category_names


def tokenize(text):

    ''' Tokenizer function that processes the message data.'''
    ''' Returns: the cleaned tokens for each message. '''

    #Replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #Normalize (lowercase and remove punctuation)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #Tokenize text
    tokens = word_tokenize(text)

    #Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens if tok not in stopwords.words("english")]

    return clean_tokens



def build_model():

    ''' Builds the ML pipeline with GridSearchCV.'''
    ''' Returns: Instantiated model. '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    parameters = {
       'vect__max_df': (0.5, 0.75, 1.0),
       'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True,False],
        'clf__n_estimators':[1,4,8,16,32,64,100],
        'clf__max_depth':range(10,100,10)
    }

    model = GridSearchCV(pipeline, param_grid = parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):


    ''' Makes a prediction and evaluates the models predictive abilities using a
        classification_report scheme.'''

    ''' Prints: Precision score (average='micro'), f1_score (average='micro'), and \
        recall_score (average='micro') for each labeled category. '''

    Y_pred = model.predict(X_test)

    #Loop through each category and prints classification report for each.
    for i in range(len(Y_pred.T)):
        cat = category_names[i]
        pred_cat = Y_pred.T[i]
        test_cat = Y_test.T[i]
        print(cat, precision_score(test_cat, pred_cat, average='micro'), f1_score(test_cat, pred_cat, average='micro'), recall_score(test_cat, pred_cat, average='micro')) )
    
    #prints final precision score over all categories
    print(precision_score(Y_pred, Y_test, average='micro'))
    print(model.best_params_)

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
