import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from pinecone_manager import initialize_pinecone, fetch_pinecone_index, load_symptom_embeddings, upsert_to_pinecone,check_index_exists_or_create,delete_index

load_dotenv(find_dotenv(), override = True)

def setup_pinecone_manager(api_key, environment, index_name, embeddings_file_path=None):
    print("Inside setup_pinecone_manager : " + api_key)
    print("Inside setup_pinecone_manager : " + environment)
    print("Inside setup_pinecone_manager : " + index_name)
    print("Inside setup_pinecone_manager : " + embeddings_file_path)
    pc = initialize_pinecone(api_key, environment)  # Initialize Pinecone
    #delete_index(pc,index_name)
    if embeddings_file_path:
        # Load embeddings and insert them into Pinecone
        symptom_df = load_symptom_embeddings(embeddings_file_path)
        # Print the type of the data and the first few rows (if it's a DataFrame)
        print(f"Type of loaded object: {type(symptom_df)}")

        # If the data is a DataFrame, print the first few rows
        if isinstance(symptom_df, pd.DataFrame):
            print(symptom_df.head())  # Display the first 5 rows
            symptom_df.to_csv('./data/check.csv')  # Save as pickle for easy reloading
        else:
            print(symptom_df)  # Print the raw data if it's not a DataFrame
        index = check_index_exists_or_create(pc, index_name,768)
        upsert_to_pinecone(index, symptom_df)
    else:
        print("Embeddings file path not provided. Skipping data insertion.")

# Main block for standalone execution
if __name__ == "__main__":
    API_KEY = os.environ.get("PINECONE_API_KEY")
    ENVIRONMENT = os.environ.get("PINECONE_ENV")
    INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
    EMBEDDINGS_FILE_PATH = "./embeddings/symptom_embeddings.pkl"

    setup_pinecone_manager(API_KEY, ENVIRONMENT, INDEX_NAME, EMBEDDINGS_FILE_PATH)
