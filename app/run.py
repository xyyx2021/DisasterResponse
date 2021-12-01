import json
import plotly
import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine
from pprint import pprint
import operator
import sys
import re
import nltk
nltk.download('stopwords')

# initializing Flask app
app = Flask(__name__)

def tokenize(text):
    
    """
    Converts a text to tokens following the pipeline below:

        * Normalize case and remove punctuations
        * split into words
        * remove stop words (English)
        * lemmatize

        Input:
        text: a string
        
        Return:
        tokenize string in a list

    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    wordsLemma = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return wordsLemma

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('classified_messages', engine)

# load model
model = joblib.load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'] 
                                                          
    genre_names = list(genre_counts.index) 
    
    # Visualize the distribution of different categories
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))

    # Extract data exclude related
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[0:10]
    categories_names = list(categories_mean.index)
    
  
    
# create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of the three Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of all Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },
        {
            'data': [
                Pie(
                    labels=category,
                    values=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categoriess',
            }
        }
    ]
     

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()