import base64
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from pinecone_manager import initialize_pinecone, fetch_pinecone_index, fetch_unique_search_item, query_pinecone  

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Setup Pinecone (Replace with your credentials and index name)
API_KEY = os.environ.get("PINECONE_API_KEY")
ENVIRONMENT = os.environ.get("PINECONE_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Streamlit App
st.title("Medical Disease/Symptoms Query Using PineCone Vector Database Search")

# Initialize Pinecone
try:
    pc = initialize_pinecone(API_KEY, ENVIRONMENT)
    #st.success("Pinecone initialized successfully!")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# Ensure Index Exists
try:
    index = fetch_pinecone_index(pc, INDEX_NAME)
    #st.success(f"Pinecone index '{INDEX_NAME}' loaded successfully!")
except Exception as e:
    st.error(f"Error fetching Pinecone index: {e}")
    st.stop()

# Fetch the list of unique diseases only after index is loaded
search_results = []

# User Input: Step 1 - Query Type
query_type = st.selectbox("Query Type", ["Select from below","Disease", "Symptom"])

choice_value = None

# Dynamically populate the second dropdown
if query_type in ["Disease" , "Symptom"]: 
    # Fetch the list of unique diseases
    try:
        search_results = fetch_unique_search_item(pc, INDEX_NAME, query_type)
        if search_results:
            # Capitalize each word in the disease names
            capitalized_search_results = [search_result.title() for search_result in search_results]
            if search_results and query_type == "Disease":  # Ensure there are diseases to show
                choice_value = st.selectbox("Select Disease", ["Select Disease"] + capitalized_search_results)
            if search_results and query_type == "Symptom":  # Ensure there are diseases to show
                choice_value = st.selectbox("Select Symptom", ["Select Symptom"] + capitalized_search_results)
            if choice_value != "Select Disease" or choice_value != "Select Symptom":
                st.write(f"You selected : {choice_value}")
        else:
            st.error("No items available to display.")
    except Exception as e:
        print(f"Error fetching search items: {e}")
        st.error("No items available to display.")
            
# Show the Search button only if a valid choice is made
if query_type in ["Disease", "Symptom"] and choice_value and choice_value != f"Select {query_type}":
    if st.button("Search"):
        if (query_type in ["Disease" , "Symptom"] and choice_value not in ["Select Disease", "Select Symptom"]):
            try:
                with st.spinner("Fetching data for selected disease/symptom..."):
                    sorted_matches = query_pinecone(pc, INDEX_NAME, choice_value)
                    if sorted_matches:
                        st.success(f"Here are all entries related to {choice_value}, sorted by similarity score:")
                        matches = sorted_matches.get("matches")
                        print(f"Matches found in Pinecone DB: {matches}")  # Debug the response structure
                        for match in matches:
                            print(f"Match is : {match}")
                            metadata = match.get("metadata", {})
                            print(f"metadata is : {metadata}")
                            score = match.get("score", "N/A")
                            print(f"score is : {score}")
                            st.write(f"**Disease:** {metadata.get('disease', 'N/A')}")
                            st.write(f"**Symptom:** {metadata.get('symptom', 'N/A')}")
                            st.write(f"**Description:** {metadata.get('description', 'No Description Available')}")
                            st.write(f"**Precautions:** {', '.join(metadata.get('precautions', []))}")
                            st.write(f"**Weight:** {metadata.get('weight', 'N/A')}")
                            st.write(f"**Similarity Score:** {score:.4f}")
                            st.markdown("---------------------------------")
                    else:
                        st.warning(f"No results found for {choice_value}.")
            except Exception as e:
                st.error(f"Error during query: {e}")
        else:
            st.error("Please select a valid option.")
