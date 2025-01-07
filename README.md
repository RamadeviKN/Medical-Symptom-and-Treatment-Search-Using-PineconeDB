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
git clone https://github.com/your-username/Medical-Symptom-Treatment-Search.git
cd Medical-Symptom-Treatment-Search

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

# Step 5: Run the Application

cd src
streamlit run app.py

Open the URL provided in the terminal (default: http://localhost:8501) in your web browser.

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

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Acknowledgements
Special thanks to the 365DataScience course Introduction to Vector Databases with Pinecone for providing a foundational understanding of vector databases and their applications. A heartfelt thank you to Elitsa Kaloyanova, the mentor for the course, for her guidance and insights that made this project possible.




