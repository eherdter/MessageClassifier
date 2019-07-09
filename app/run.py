import json
import plotly
import pandas as pd
import boto3
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

#load model from S3
#s3 = boto3.client("s3", region_name="us-east-2")
#response = s3.get_object(Bucket="eherdterprojects", Key="classifier.pkl")

#body_string = response['Body'].read()
#model = pickle.loads(body_string)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #make plot of messages with multiple labels
    #messages having multiple labels
    col_list= list(df.drop(['direct_report','message','original','genre'], axis=1))
    df['mult_lab_sum'] = df[col_list].sum(axis=1)

    df['mult_lab_sum'] = df['mult_lab_sum'].astype(str)
    values=df.mult_lab_sum.value_counts().values
    num_labels=df.mult_lab_sum.value_counts().index.values

    #make plot of #comments that have each label
    nrows_entire = df.shape[0]
    def label_by_gen(df1):
        df_cols = df1.drop(['direct_report','message','original','genre'], axis=1)
        df_bkdown = pd.DataFrame(df_cols.sum()/nrows_entire, columns=['percent']).sort_values(by=['percent'])
        df_labels = df_bkdown.index.tolist()
        df_perc = df_bkdown['percent'].values
        return df_labels, df_perc

    news = df[df['genre'] == 'news']
    news_labels, news_perc = label_by_gen(news)

    social = df[df['genre'] == 'social']
    social_labels, social_perc = label_by_gen(social)

    direct = df[df['genre'] == 'direct']
    direct_labels, direct_perc = label_by_gen(direct)




    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hoverinfo='label+percent', textinfo='value',
                    textfont=dict(size=20),
                    marker=dict(colors=['#FEBFB3', '#E1396C', '#96D38C'],
                    line=dict(color='#000000', width=2))
                )
                ],
            'layout' : {
                'title': 'Distribution of Message Genres'}
        },
        { 'data': [Bar(
                    x=news_perc,
                    y=news_labels,
                    name='news',
                    orientation = 'h',
                    marker = dict(color='#E1396C',
                                  line=dict( color='#000000',
                                            width=1.5))
                    ),

                    Bar(
                    x=direct_perc,
                    y=direct_labels,
                    name='direct',
                    orientation = 'h',
                    marker = dict(color='#febfb3',line=dict( color='#000000',
                                            width=1.5))
                    ),

                    Bar(
                    x=social_perc,
                    y=social_labels,
                    name='social',
                    orientation = 'h',
                    marker = dict(color='#96D38C', line=dict( color='#000000',
                                            width=1.5))
                    )

                  ],
           'layout': {
               'title': 'Distribution of Assigned Labels by Message Genre',
               'yaxis': {
                   'automargin': True
               },
               'barmode': 'stack'
           }
        },

        {
            'data': [
                Bar(
                    x=num_labels,
                    y=values,
                    marker=dict(color='#9be8d1', line=dict( color='#6aebc4',
                                            width=1.5))
                )
            ],

            'layout': {
                'title': 'Distribution of Number of Labels Assigned to Each Message',
                'yaxis': {
                    'title': "Sum"
                },
                'xaxis': {
                    'title': "Number of Assigned Labels"
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
    classification_results = dict(zip(df.columns[:35], classification_labels))


    # This will render the go.html Please see that file.

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
