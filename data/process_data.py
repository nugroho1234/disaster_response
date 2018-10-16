import pandas as pd
import numpy as np
import argparse
from sqlalchemy import create_engine

#parsing for terminal input
parser = argparse.ArgumentParser()
parser.add_argument('--messages', type = str, help = 'input messages csv file', default = 'disaster_messages.csv')
parser.add_argument('--categories', type = str, help = 'input categories csv file', default = 'disaster_categories.csv')
parser.add_argument('--database', type = str, help = 'input target database', default = 'DisasterResponse.db')
args = parser.parse_args()

#loading the csv files and database file
if args.messages:
    messages_csv = args.messages
if args.categories:
    categories_csv = args.categories
if args.database:
    database = args.database

#loading messages and categories dataframe
messages = pd.read_csv(messages_csv)
categories = pd.read_csv(categories_csv)

#merging messages and categories into a single dataframe
#df = pd.merge(messages,categories, on = 'id', how = 'inner')
categories.drop(categories.columns[categories.columns.isin(messages.columns)],axis=1,inplace=True)
df = pd.concat([messages, categories], axis = 1)

#preparing categories dataframe
categories = df['categories'].str.split(';', expand = True)
#creating column name and replace categories columns with this
row = categories.iloc[0].tolist()
category_colnames = [x[:-2] for x in row]
categories.columns = category_colnames
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.extract('(\d)')

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

#replace original categories column with the prepared categories column
df.drop('categories', axis = 1, inplace = True)
df = pd.concat([df,categories], axis = 1)

#dropping duplicates
df.drop_duplicates(inplace = True)

if args.database:
    database = args.database

engine_path = 'sqlite:///' + database

engine = create_engine(engine_path)
df.to_sql('disaster_response', engine, index=False, if_exists = 'replace')
