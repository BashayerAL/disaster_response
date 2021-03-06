import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

#loading data
def load_data(messages_filepath, categories_filepath):
     messages = pd.read_csv(messages_filepath)
     categories = pd.read_csv(categories_filepath)
     df = messages.merge(categories, on = ['id'])
     return df

#cleaning data
def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames
    
    for column in categories:
         # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
          #for each column category convert it to int
        categories[column] = categories[column].astype(int)
    
    #drop column category from df
    df.drop('categories', axis = 1, inplace = True)
    
   #merge the df and categories dataset
    df = pd.concat([df, categories], axis = 1)
    df.related.replace(2, 1, inplace=True)
    
    #drop any duplicates value
    df.drop_duplicates(inplace = True)
    print(df.shape)
    #return df 
    return df
#function for saving data
def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///disasterdb.db',if_exists='replace')
    #convert the data to sql
    df.to_sql('disaster', engine, index=False)
    


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
