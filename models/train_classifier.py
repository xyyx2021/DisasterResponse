import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import numpy as np
import pandas as pd
import re
import pickle
import nltk
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
sys.path.append('.')

import numpy as np
import re



def load_data(database_filepath):
    
    """
    Loads data from SQL Database

    Input:
    database_filepath: SQL database file

    Return:
    X panda dataframe: Features dataframe
    Y panda dataframe: Target dataframe
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql('SELECT * FROM classified_messages', engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    
    # mapping extra values to `1`
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    
    """
        Converts a text to tokens following the pipeline below:

        * Normalize case and remove punctuations
        * split into words
        * remove stop words (English)
        * lemmatize
        * stems

        Input:
        text: a string
        stop_words: list of stop words. Default = stopwords.word("english") from nltk

        Return:
        tokenize string in a list

        """

    # prep nltk transformation objects

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words=stopwords.words("english")
    lemmed = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word not in stop_words]

    # Reduce words to their stems
    stemmed = [stemmer.stem(word) for word in lemmed]

    return stemmed



def build_model():
    
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing grid search
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=24)))
        ])
    
    # parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__learning_rate': [1.0, 2.0]
        }

   # Model from grid search
    gscv = GridSearchCV(pipeline, param_grid=parameters,  verbose=2, cv=3)

    return gscv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Shows model's average performance on test data using defined functions above

    Input:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    
    Return:
    none
    """

    # calculate model predictions
    Y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == Y_pred)))

def save_model(model, model_filepath):
    
    """
    Saves the model to a pickle file    
    
    Inputs:
    model: Trained model
    model_filepath: Filepath to save the model
    
    Return:
    none
    """

    pickle.dump(model, open(model_filepath, 'wb'))




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