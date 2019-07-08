# Classifying message data for rapid disaster response

### About
#### ETL and ML Pipeline Methods
A message classification model was trained using messages sent during natural disasters around the globe. The data set contains 25,000+ messages sent during events including the 2010 earthquakes in Haiti and Chile, the devastating floods in Pakistan in 2010, and super-storm Sandy. In addition, additional news articles spanning a large number of years and other different disasters are included in the data set.The data have been anonymized and encoded with 36 different categories related to disaster response.The data were provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). 

The data were cleaned and transformed in an ETL pipeline prior to being fed into a natural language processing ML pipeline. A RandomForestClassifier equipped to perform multi-output classification was trained using GridSearchCV (with 5-fold Cross Validation).

#### Flask Web Application
The trained classifier is used within a Flask web application where users can insert a new message for rapid classification. The web app is currently deployed on an Amazon EC2 computing instance view-able here *******

However, if you wish to produce your own trained classifier follow the below instructions:

### Instructions

To produce your own trained classifier:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Import Considerations

This data set presents itself as a very intensive multi-label classification task as each message may be categorized in up to 36 different categories. In addition, many categories are extremely imbalanced. For example, there are only XX% of the messages that are labeled within the ## category. 

Due to such extreme imbalance, it is strongly advised against using accuracy but rather using a metric such as precision or f1_score. There are some methods to address the imbalance specifically within multi-label data sets including random resampling algorithms which are documented in the literature, for example see [Charte et al. 2015](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1790_2015-Neuro-Charte-MultiLabel_Imbalanced.pdf). 

Additionally, there are some methods built within or on top of sklearn that may address such imbalance including [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and [scikit-multilearn](http://scikit.ml/api/skmultilearn.adapt.mlknn.html). 

However, each have their drawbacks. For example, **BalancedRandonForestClassification** does not currently support resampling for multi-label data sets. Therefore, the current classifier has room for improvement. Some immediate suggestions for improvement may be to select specific categories of interest and manually oversample the messages within the category. 

### Acknowledgements 
The skeleton files and data set for this project were provided by [Udacity](https://www.udacity.com/). 

