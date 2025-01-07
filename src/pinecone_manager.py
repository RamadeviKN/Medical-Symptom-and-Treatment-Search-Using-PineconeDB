import pinecone
import pandas as pd
import ast

from pinecone import Pinecone, Index, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Global variable for Pinecone index
index = None

# Initialize Pinecone environment
def initialize_pinecone(api_key: str, environment: str)-> Pinecone:
    try:
        pc = Pinecone(api_key=api_key, environment=environment)
        print("Pinecone initialized successfully.")
        return pc
    except Exception as e:
        raise RuntimeError(f"Error initializing Pinecone: {e}")
    
# Check or Create Index
def check_index_exists_or_create(pc, index_name: str, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1")->Index:
    try:
        print("-----"+ index_name + "----------")
        existing_indexes = pc.list_indexes()
        # Debugging: Print the list of indexes and the index_name
        print(f"Existing indexes: {existing_indexes}")
        # Check if the index exists
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists.")
        else:
            print("Inside else loop")
            # Create a new index if it doesn't exist
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(f"Index '{index_name}' successfully created.")
        # Return the index object
        return pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"Error ensuring index exists: {e}")

# Create or connect to a Pinecone index
def fetch_pinecone_index(pc, index_name: str)->Index:
    try:
        print("-----"+ index_name + "----------")
        existing_indexes = pc.list_indexes()
        # Debugging: Print the list of indexes and the index_name
        print(f"Existing indexes: {existing_indexes}")
        #if index_name not in existing_indexes:
           #raise ValueError(f"Pinecone index '{index_name}' does not exist.")
        return pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"Error fetching Pinecone index: {e}")

# Load embeddings data (conditionally)
def load_symptom_embeddings(file_path: str):
    try:
        data = pd.read_pickle(file_path)
        print(f"Loaded data of type: {type(data)}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

# Insert data into Pinecone
def upsert_to_pinecone(medical_symptoms_index, symptom_df):
    try:
        # Prepare vectors for upsert
        vectors_to_upsert = [
        (
            row["unique_id"], 
            row["embedding"].tolist(), 
            {
                "disease": row['disease'],
                "symptom": row['symptom'],
                "description": row['description'],
                "precautions": ast.literal_eval(row['precautions']) if isinstance(row['precautions'], str) else row['precautions'],
                "weight": row['weight']
            }
        )
    for index, row in symptom_df.iterrows()
    ]
        
        # Perform upsert operation
        medical_symptoms_index.upsert(vectors=vectors_to_upsert)
        print("Data successfully upserted to Pinecone index.")  
    except KeyError as e:
        print(f"KeyError: Missing expected column in symptom_df: {e}")
    except AttributeError as e:
        print(f"AttributeError: Issue with symptom_df format or content: {e}")
    except ValueError as e:
        print(f"ValueError: Problem with data in symptom_df: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Query Pinecone Function
def query_pinecone(pc, index_name, query, top_k=10):
    """
    Query Pinecone index using an input query.
    
    Args:
        query (str): The query string (e.g., symptom or disease name).
        top_k (int): Number of top results to retrieve.
        
    Returns:
        dict: Pinecone query results including metadata.
    """
    medical_symptoms_index = pc.Index(index_name)
    # Initialize the model for querying
    model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    try:
        # Generate query embedding
        query_embedding = model.encode(query, show_progress_bar=False).tolist()
        
        # Query Pinecone
        results = medical_symptoms_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        print(f"Query Response: {results}")  # Debug the response structure
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None
    
# Method to delete an index by name
def delete_index(pc, index_name: str):
    try:
        # Check if the index exists
        if index_name in pc.list_indexes():
            # Delete the index if it exists
            pc.delete_index(index_name)
            print(f"Index '{index_name}' successfully deleted.")
        else:
            print(f"Index '{index_name}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"Error deleting index '{index_name}': {e}")
    
# Function to fetch unique diseases
def fetch_unique_search_item(pc, index_name, search_item, top_k=100):
    try:
        # Get the Pinecone index object
        index = pc.Index(index_name)

        # Generate a dummy query vector (it could be zeros or random)
        query_vector = [0] * 768  # Replace 1536 with your vector dimension
        
        # Perform the query with dummy vector
        query_response = index.query(index_name=index_name, top_k=top_k, vector=query_vector, include_metadata=True)
        
        # Extract metadata from the query response and get unique diseases
        search_items = set()
        for match in query_response["matches"]:
            metadata = match["metadata"]
            if metadata:
                search_result = metadata.get(search_item.lower())  # Get the disease field from metadata
                if search_result:
                    search_items.add(search_result)
        
        return list(search_items)
    
    except Exception as e:
        raise RuntimeError(f"Error fetching unique diseases: {e}")
    
def fetch_disease_vector(pc, index_name, disease_name):
    try:
        # Get the Pinecone index object
        index = pc.Index(index_name)

        # Query Pinecone to get the vector for the selected disease
        query_response = index.query(index_name=index_name, top_k=1, filter={"disease": disease_name}, include_values=True)

        if query_response and "matches" in query_response and len(query_response["matches"]) > 0:
            # Extract vector from the response
            vector = query_response["matches"][0]["values"]
            return vector
        else:
            return None
    except Exception as e:
        raise RuntimeError(f"Error fetching vector for disease {disease_name}: {e}")
        

            




