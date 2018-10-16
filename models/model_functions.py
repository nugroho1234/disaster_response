from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier


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

        if count > 0:
            return True
        else:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply keyword_search function to all values in X
        X_tagged = pd.Series(X).apply(self.keyword_search)

        return pd.DataFrame(X_tagged)


def tokenize(text):
    '''
    Input: text to be tokenized
    Output: clean tokens
    '''
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in token:
        clean_tokens.append(lemmatizer.lemmatize(tok))
    return clean_tokens

def build_pipeline():
    '''
    Input: none
    Output: pipeline model with feature union of nlp_pipeline (vect and tfidf)
    and KeywordExtractor class. The classifier is AdaBoostClassifier wrapped in
    MultiOutputClassifier
    '''
    pipeline_2 = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

        ('keyword', KeywordExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline_2

def grid_search(pipeline, X_train, X_test, y_train, y_test, parameters):
    '''
    Inputs:
    pipeline: ML pipeline
    X_train, X_test, y_train, y_test: dataframe splitted into train and test sets
    parameters: GridSearchCV parameters

    Outputs grid search model and y_pred in the form of a dataframe  
    '''
    cv = GridSearchCV(estimator = pipeline, param_grid = parameters, verbose = 2)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = y_test.columns
    return cv, y_pred_df
