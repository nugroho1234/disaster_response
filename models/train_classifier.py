# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import model_functions as mf
from model_functions import KeywordExtractor
import argparse

#parsing for terminal input
parser = argparse.ArgumentParser()
parser.add_argument('--database', type = str, help = 'input target database', default = 'DisasterResponse.db')
parser.add_argument('--model', type = str, help = 'input target model to save', default = 'classifier.pkl')
args = parser.parse_args()

if args.database:
    database = args.database
if args.model:
    model = args.model

#TODO: modularize code, create docstring
database_engine = 'sqlite:///' + database

engine = create_engine(database_engine)
df = pd.read_sql_table('disaster_response', engine)
X = df['message']
y = df[df.columns[4:]]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

parameters = {
        'features__nlp_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__nlp_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__nlp_pipeline__vect__max_features': (None, 5000, 10000),
        'features__nlp_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200]
        }

pipeline = mf.build_pipeline()

cv, y_pred_df = mf.grid_search(pipeline, X_train, X_test, y_train, y_test, parameters)

for col in y_test.columns:
    print(col,'\n', classification_report(y_test[col], y_pred_df[col]))

joblib.dump(cv, model)
