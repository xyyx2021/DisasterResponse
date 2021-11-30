# Disaster Response Pipeline Project
Repository for the Disaster Respose Pipeline Project as part of the Udacity Nanodegree Data Science. 

## Project Overview
The motivation of this project is to classify disaster messages. Model was built based on analyzing the disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a>.

This project has three components: processing data, building model and web app where the user can input a new message and get the classification results, with visualization of the data.

## Project Components
More details abot the three components in this repository as below:

1. ETL pipeline to process data
* Load the messages and their categories
* Merge the two datasets
* Clean the dataset
* Save it in a SQLite database

2. ML Pipeline to build model
* Load data from the SQLite database
* Split the dataset into training and test sets
* Build machine learning pipeline 
* Train and optimize a model using GridSearchCV
* Evaluate the results from prediction
* Export the final model as a pickle file

3. Flask Web App to classify the message

* User friendly web interface to classify any new message from user's input


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Test the web app at https://view6914b2f4-3001.udacity-student-workspaces.com/
