import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    loads messages and categories from csv file, merge them into one dataframe

    Input:
    messages_file_path str: messages CSV file
    categories_file_path str: categories CSV file

    Output:
    merged_df pandas_dataframe
    """

    # load raw data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge on id
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    
    """
    - Cleans the merged dataframe to prepare for the ML model
    
    Input:
    Merged dataframe returned from load_data() 

    Output:
    Cleaned data for the ML model
    """
    
    # Split categories into separate category columns
    categories = df['categories'].str.split(";", expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[1, :]

    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

        # coerce anything greater than 1 into ==1
        categories.loc[categories[column] > 1, column] = 1

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    
    """
    Saves the cleaned data to an SQL database

    Input:
    df: Cleaned data returned from clean_data() function
    database_file_name: File path of SQL Database 

    Return:
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename)) 
    df.to_sql('classified_messages', engine, index_label='id', index=False, if_exists='replace')

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