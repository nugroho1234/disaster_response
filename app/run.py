import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class KeywordExtractor(BaseEstimator, TransformerMixin):

    def keyword_search(self, text):
        # tokenize by sentences
        word_list = word_tokenize(text)
        type_list = ['water', 'thirsty', 'food', 'hunger', 'hungry', 'medic', 'medical', 'medication']
        count = 0
        for word in word_list:
            for typ in type_list:
                if word in typ:
                    count += 1
                else:
                    count += 0
            # return true if the first word is an appropriate verb or RT for retweet
        if count > 0:
            return True
        else:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.keyword_search)

        return pd.DataFrame(X_tagged)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #df['sum_row'] = df[df.columns[5:]].sum(axis = 1)
    #request_by_genre = df.groupby('genre').count()[df[df.columns[5:]].sum(axis = 1)]
    index_req = ['Food', 'Water', 'Shelter']
    vals_req = [df.food.sum(), df.water.sum(), df.shelter.sum()]
    #df.drop('sum_row', axis = 1, inplace = True)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
                    x=index_req,
                    y=vals_req
                )
            ],

            'layout': {
                'title': 'Requests of Food, Water, and Shelter',
                'yaxis': {
                    'title': "Label"
                },
                'xaxis': {
                    'title': "Count"
                }
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
