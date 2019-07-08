# Classifying message data for rapid disaster response

### About
A message classification model was trained using messages sent during natural disasters around the globe. The dataset contains 25,000+ messages sent during events including the 2010 earthquakes in Haiti and Chile, the devestating floods in Pakistan in 2010, and super-storm Sandy. In addition, additional news articles spanning a large number of years andother different disasters are included in the dataset.The data have been anonimized and encoded with 36 different categories related to disaster response.The data were provided by [Figure Eight]("https://www.figure-eight.com/dataset/combined-disaster-response-data/"). 

The data were cleaned and transformed in an ETL pipeline prior to being fed into a natural language processing ML pipeline. A RandomForestClassifier equipped to deal with multi-label data was trained using GridSearchCV (with 5-fold Cross Validation).


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
