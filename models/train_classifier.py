import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def load_data(database_filepath):
    
    ''' Loads data from database.'''
    ''' Returns: X, Y, and category names. '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    X= df.message.values
    Y = df.iloc[:, :36].values
    categories = df.iloc[:, :36].columns
    
    return X, Y, categories


def tokenize(text):
    
    ''' Tokenizer function that processes the message data.'''
    ''' Returns: the cleaned tokens for each message. '''
    
    #replace urls
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #tokenize the text
    tokens = word_tokenize(text)
    
    #instantiate the WordNetLammatizer
    lemmatizer = WordNetLemmatizer()
    
    #lemmatize the tokens from above
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens
    

def build_model():
    
    ''' Builds the ML pipleline and performs Grid Search with CV.'''
    ''' Returns: Instantiated model that should be fit in a following function. '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    
    parameters = {
    'clf__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
