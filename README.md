# Medical-Symptom-and-Treatment-Search-Using-PineconeDB
A web-based application that allows users to search for medical symptoms and their associated diseases, treatments, and precautions using a powerful PineconeDB-backed vector similarity search. This project leverages machine learning models, Python, and modern web development frameworks to provide an intuitive and effective user experience.


# Features
Search for Diseases by Symptoms: Input symptoms to get detailed information about possible diseases.
Precautionary Measures: Provides recommendations for treatment and precautions.
Efficient Vector Search: Uses PineconeDB for storing and retrieving high-dimensional embeddings.
Lightweight UI: Built with Streamlit for a simple and interactive web interface.
Scalable: Designed for easy deployment and scalability.

# Who Can Use This App?
This application is designed for a wide range of users:

Healthcare Professionals: Quickly reference potential diagnoses and precautions based on symptoms.
Medical Students: Enhance learning by exploring symptom-disease relationships interactively.
General Public: Get insights into common symptoms and precautionary measures, empowering self-care.
Health Researchers: Use the app as a prototype for understanding how vector databases can be applied to medical data.

# Applications Across Fields
Healthcare: To assist doctors and nurses in providing quick information on symptoms and treatments.
Education: As a learning tool for students in medical and data science fields.
Health Technology: A potential foundation for building advanced medical diagnostic tools.
Research and Development: To test and evaluate the utility of vector databases in medical use cases.

# Setup Instructions

Prerequisites
Python 3.8 or higher
A Pinecone account and API key
Git installed on your machine

# Step 1: Clone the Repository
git clone [https://github.com/your-username/Medical-Symptom-Treatment-Search.git](https://github.com/RamadeviKN/Medical-Symptom-and-Treatment-Search-Using-PineconeDB.git)
cd Medical-Symptom-Treatment-Search-Using-PineconeDB

# Step 2: Set Up Virtual Environment
python -m venv .venv
source .venv/bin/activate    # On Linux/Mac
.venv\Scripts\activate       # On Windows

# Step 3: Install Dependencies
pip install -r requirements.txt

# Step 4: Configure Environment Variables
1. Create a .env file in the root directory:
touch .env

2. Add the following:
PINECONE_API_KEY=<Your Pinecone API Key>
PINECONE_ENV="gcp-starter"
PINECONE_INDEX_NAME=<Your Index Name inside double quotes>

# Step 5: Prepare the Data
Ensure all data extracts (CSV files) are placed inside the /data folder.
1. disease_symptoms.csv
2. symptom_description.csv
3. symptom_precaution.csv
4. symptom_severity.csv

# Step 6: Perform Exploratory Data Analysis (EDA)
Open and run the notebook /src/data_preprocessing.ipynb.

This step will:
1. Concatenate the CSV files in the /data folder.
2. Clean and explore data with missing values
3. Generate a combined preprocessed file: preprocessed_symptom_dataset.csv.

# Step 7: Generate Embeddings
Open and run the notebook /src/embedding_generation.ipynb.

This step will:
1. Generate embeddings from the data in preprocessed_symptom_dataset.csv.
2. Save the embeddings to symptom_embeddings.pkl inside the /embeddings folder.

# Step 8: Upsert Data into Pinecone Database
Run the script /src/data_upsertion.py.

This step will:
1. Create a Pinecone index named medical-symptoms-index.
2. Insert data on diseases, symptoms, and precautions into the index.
Note:
1. Log in to Pinecone and navigate to the Database/Indexes section to validate the created index.
2. An API key is required to perform this step.

# Step 9: Launch the Streamlit App
After completing the above steps and validating the index in Pinecone Database, navigate to the /src folder.
Run the following command:

streamlit run app.py
Open the URL provided in the terminal (default: http://localhost:8501) in your web browser.

This will load the user interface. You can:
1. Query the database using a disease or symptom.
2. Retrieve semantically similar results powered by the vector database, showcasing the advantage of storing high-dimensional data in vector form.

# Usage
Search by Disease or Symptom: Select a disease or symptom (e.g., "fever") in the input box and press Search button.
View Results: Get a list of related diseases, symptoms, descriptions, precautions, and similarity scores retrieved from Pinecone index.
Interactive UI: Navigate easily through the application for a seamless user experience.

# Technologies Used
Python: Core programming language.
Streamlit: For creating the user interface.
PineconeDB: Vector database for similarity search.
SentenceTransformer: To encode disease and symptom descriptions into embeddings.

# Future Enhancements
1. Add a feature to handle multiple symptoms for more accurate disease prediction.
2. Enable export of search results to CSV or PDF.
3. Integrate additional ML models for better recommendations.
4. Add support for multilingual search.

# Demo
To see how the application works, check out the demo video inside the repo folder "demo":  
[Download Demo Video](demo/medical-symptom-and-treatment-srarch-using-pineconedb-video.mp4) 

Pinecone Database Index View:
[Download Demo Video](demo/pinecone-database-index-view.mp4) 

![image](https://github.com/user-attachments/assets/1b92b375-b5dc-4c28-8507-e1016a1b38d4)

![image](https://github.com/user-attachments/assets/7c58ce66-c8a5-46f2-9dc6-af97d548c2fd)

![image](https://github.com/user-attachments/assets/942d3446-279d-46ba-99fc-d23f81786a60)

![image](https://github.com/user-attachments/assets/6d4aab50-6a71-4d91-be2a-6f8013434ab5)

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Acknowledgements
Special thanks to the 365DataScience course Introduction to Vector Databases with Pinecone for providing a foundational understanding of vector databases and their applications. A heartfelt thank you to Elitsa Kaloyanova, the mentor for the course, for her guidance and insights that made this project possible.




