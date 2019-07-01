import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    ''' Loads messages and category datasets and joins them 
    together using the common id.'''
    
    ''' Returns: pandas.DataFrame '''
    
    #Load data.
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    
    #Merge datasets using common id. 
    df = categories.set_index('id').join(messages.set_index('id'))
    
    return df
    


def clean_data(df):
    
    '''Tidy and clean df. Tidies the categories columns, cleans up duplicates rows,
    removes rows with non binary (0/1) category entries, removes rows where there is no category selected (1).'''
    
    '''Returns: pandas.DataFrame '''
    
    #split categories into separate category columns and assign column names
    categories = df.categories.str.split(';', expand=True)
    categories.columns = categories.iloc[0].apply(lambda x: x[:len(x)-2])
    
    #convert category values to just binary (0/1)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    
    #replace categories column in df with new categories
    df = df.drop(columns=['categories'])
    df = pd.concat([categories,df], axis=1)
    
    #removes duplicate rows
    df = df.drop_duplicates()
    
    #removes rows where related column is non-binary (e.g. ! (0|1))
    df = df[df['related'] !=2]
    
    #remove rows where there is not selected category marked
        #make a new sum column to check
    col_list= list(df.drop([ 'direct_report','message','original','genre'], axis=1))
    df['sum'] = df[col_list].sum(axis=1)
    
    df = df[df['sum'] !=0]
        #drop the sum column
    df.drop(['sum'], axis=1, inplace=True)
    
    #remove rows with message starting with "NOTES:"
    df = df[~df.message.str.contains("NOTES:")]
    
    return df
    


def save_data(df, database_filepath):
    
    ''' Saves cleaned df to a SQL database.'''
    
    engine = create_engine('sqlite:///' + database_filepath)
    
    df.to_sql('messages', con=engine, if_exists='replace', index=False)
    



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
