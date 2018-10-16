# Disaster Response Project
This is a data science project which has an aim to classify text messages into what people need during a disaster.

# Installation
Clone this repo: https://github.com/nugroho1234/disaster_response.git
I used python 3.6 to create this project and the libraries I used are:
1. Pandas
2. Numpy
3. Flask
4. Plotly
5. Scikit-learn
6. Argparse
7. Sqlalchemy

# Project Motivation
Natural disaster is something which is classified as force majeure. Not every city / country has adequate resource to deal with this. 
As of now, handling disaster victims is quite difficult since usually communications are cut. However, as soon as it's up, people can post
on social media about their condition. Therefore, it will be real nice if data science can help classify natural disaster's victims' text. 
It will help organizations to map which area needs which resources the most.

# File Descriptions
### ETL Pipeline Preparation.ipynb
This is the notebook I used to prepare my ETL Pipeline. It merges the message and category into one dataframe, cleans it, and store it into a dataabase.
### ML Pipeline Preparation.ipynb
This is the notebook I uesd to prepare my ML Pipeline. It uses ML pipeline with NLP vectorizer, transformation, and custom transformer, with AdaBoostClassifier as the classifier.
It also uses grid search to optimize the pipeline. As a warning, running the fitting part can take a long time.
### process_data.py
This is the .py file for the ETL Pipeline. To run: go to command line and type:
python process_data.py
The arguments for process data is:
--messages (input the csv file of messages)
--categories (input the csv file of categories)
--database (input the target database to save)
### functions.py
These are functions and a keywordextractor class for train_classifier.py
### train_classifier.py
This is the .py file for ML Pipeline. To run: go to command line and type:
python train_classifier.py
The arguments are:
--database (target database to load the data from)
--modl (model file name to save)
### run.py
This is the app file. The app will get the model and database from both train_classifier and process_data.py and display it in a web page.
To run:
1. go to command line and type python run.py
2. go to your browser and type localhost:3001

# How to use this project
This project is interactive in nature. Type the messages in the form, and click classify. It will classify the message into one or more categories such as food, water, or shelter.

# Limitation
There are imbalances in the data. For example, the water category has few examples compared to the others. I tried to tackle this problem using KeywordExtractor, a custom transformer I built. However, the result is still not satisfactory. I managed to classify thirsty to water, but even if I type water, it will classify the message as aid.

