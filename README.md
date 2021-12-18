# The Disaster_Response


Project Problem:

The data contained 30k real messages received during disasters. 
Each message is labeled with its category,Thereforeit's a multilabel classifier problem.

Project structure:

 1.Data Processing:  ETL (Extract, Transform, and Load) pipeline that processes messages and category data from CSV file.
 
 2.Machine Learning Pipeline: Creating an ML pipeline that uses NLTK and GridSearchCV to output a final model that predicts message classifications 
 for the 36 categories.(Multi_Classification)
 
 3.Web development: The final results will be in a Flask web app that classifies messages in real time.
 
 # Disaster Response Pipeline Project
 
 The Installation Required:

Python 3.5+ (I used Python 3.7)
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Web App and Data Visualization: Flask, Plotly

# Acknowledgements
-Udacity for providing the Data Science Nanodegree Program.
-Figure Eight for providing messages dataset to train my model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to
https://view6914b2f4-3001.udacity-student-workspaces.com

 
# Screenshots:

 ![1](https://user-images.githubusercontent.com/27586577/146645913-36f5787e-5e8f-40b2-97c9-7602cecf7818.png)

![2](https://user-images.githubusercontent.com/27586577/146645924-717c152e-1631-486d-9ce9-a05efb98dc89.png)
