from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Load environment variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# Initialize Qdrant client
qdrant_client = QdrantClient(qdrant_url, api_key=qdrant_key)

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Create Qdrant vector store with embeddings
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embed_model  # ✅ Pass embedding model
)

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)

@app.route('/')
def index():
    return '<h1>Welcome to Matthew Legal Assistant</h1>'

@app.route('/upload', methods=['POST'])
def upload_file():
    """Uploads a PDF, extracts text, generates embeddings, and stores them in Qdrant."""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)

        # Load PDF and extract text
        loader = PyPDFLoader(temp_file_path, extract_images=False)
        pages = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        doc_list = [
            Document(page_content=pg, metadata={"page_no": i + 1})
            for i, page in enumerate(pages)
            for pg in text_splitter.split_text(page.page_content)
        ]

        # Store embeddings in Qdrant
        qdrant.add_documents(doc_list)

        # Clean up temporary files
        os.remove(temp_file_path)
        os.rmdir(temp_dir)

        return jsonify({"message": "File uploaded and embeddings stored successfully", "chunks_stored": len(doc_list)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    """Handles user queries by retrieving relevant embeddings from Qdrant and responding using GPT."""
    user_data = request.get_json()
    query = user_data.get("question")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Retrieve relevant documents from Qdrant
        relevant_docs = qdrant.similarity_search(query, limit=3)

        if not relevant_docs:
            return jsonify({"response": "No relevant information found in the document."})

        # Prepare context for GPT
        context = "\n\n".join([f"Page {doc.metadata['page_no']}:\n{doc.page_content}" for doc in relevant_docs])
        formatted_prompt = f"""
        You are a professional legal assistant. Provide a clear, concise, and accurate response to the user's question based on the context provided. 

        Context:
        {context}

        Question:
        {query}

        Answer should be to the point and summarized.
        """

        # Get response from GPT
        response = llm(formatted_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        return jsonify({
            "response": response_text.strip(),
            "retrieved_documents": [doc.page_content for doc in relevant_docs],
            "formatted_prompt": formatted_prompt,
        })

    except Exception as e:
        return jsonify({"error": "An error occurred while processing your request.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, request, jsonify
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from langchain_qdrant import QdrantVectorStore
# from langchain_openai import ChatOpenAI
# from flask_cors import CORS
# import tempfile
# import os
# import re

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# # Text-cleaning function to improve OCR output
# def clean_text(text):
#     # Example text-cleaning steps
#     text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
#     text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
#     return text.strip()

# # Load and preprocess PDF
# print("Loading Data...")
# try:
#     loader = PyPDFLoader("C:/Users/Umar/Downloads/Baldwins OCR.pdf", extract_images=False)
#     pages = loader.load()
#     print("Data loaded successfully")
# except Exception as e:
#     print(f"Error loading data: {e}")
#     pages = []

# # Clean text and split documents
# if pages:
#     r_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
#     doc_list = [
#         Document(page_content=clean_text(pg), metadata={"page_no": i + 1})
#         for i, page in enumerate(pages)
#         for pg in r_splitter.split_text(page.page_content)
#     ]
# else:
#     doc_list = []

# # Set up the embedding model and Qdrant vector store
# embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# qdrant_url = "https://bfdbdbbd-6a7e-48a7-83dc-8725edd588d3.us-east4-0.gcp.cloud.qdrant.io:6333"
# qdrant_key = "0ETKHECLieM7k-umC6Zec_eY1iFWAbugNDRBd76Ly-goAWfr54_6RA"
# collection_name = "Mishak_Lawbot_Baldwin"

# try:
#     qdrant = QdrantVectorStore.from_documents(
#         doc_list,
#         embed_model,
#         url=qdrant_url,
#         prefer_grpc=True,
#         api_key=qdrant_key,
#         collection_name=collection_name,
#     )
# except Exception as e:
#     print(f"Error initializing Qdrant: {e}")

# # Initialize the language model
# openai_api_key = "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"
# llm_name = "gpt-4o-mini"
# llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai_api_key, temperature=0)

# prompt_template = """
# Answer the user's question using only the following context extracted from the document.

# Context:
# {context}

# Question:
# {question}
# """

# # Define routes
# @app.route("/")
# def index():
#     return "<h1>Welcome to Matthew Legal Assistant</h1>"

# @app.route("/upload", methods=["POST"])
# def upload_file():
#     try:
#         # Get file from the request
#         file = request.files.get("file")
#         if not file:
#             return jsonify({"error": "No file provided"}), 400

#         # Save file temporarily
#         temp_dir = tempfile.mkdtemp()
#         temp_file_path = os.path.join(temp_dir, file.filename)
#         file.save(temp_file_path)

#         # Load and extract text from the document
#         loader = PyPDFLoader(temp_file_path, extract_images=True)
#         pages = loader.load()

#         # Clean and combine text
#         document_text = "\n".join([clean_text(page.page_content) for page in pages])

#         # Create prompt
#         prompt = f"Please summarize the following document:\n{document_text}"

#         # Get response from GPT
#         response = llm(prompt)
#         response_text = response.content if hasattr(response, "content") else str(response)

#         return jsonify({"response": response_text.strip()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_data = request.get_json()
#     query = user_data.get("question")

#     if not query:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         # Retrieve relevant documents from Qdrant
#         relevant_docs = qdrant.similarity_search(query, limit=3)
#         if not relevant_docs:
#             return jsonify({"response": "No relevant information found in the document."})

#         # Prepare context and format prompt
#         context = "\n\n".join(
#             [f"Page {doc.metadata['page_no']}:\n{doc.page_content}" for doc in relevant_docs]
#         )
#         formatted_prompt = prompt_template.format(question=query, context=context)

#         # Get response from the language model
#         response = llm(formatted_prompt)
#         response_text = response.content if hasattr(response, "content") else str(response)

#         return jsonify({
#             "response": response_text.strip(),
#             "retrieved_documents": [doc.page_content for doc in relevant_docs],
#             "formatted_prompt": formatted_prompt,
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from langchain_qdrant import QdrantVectorStore
# from langchain_openai import ChatOpenAI
# from flask_cors import CORS
# import tempfile
# import os

# # Initialize Flask app

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# # Load and preprocess PDF
# print("Loading Data...")
# loader = PyPDFLoader(
#  "C:/Users/Umar/Downloads/Legal Book Law.pdf",
#     extract_images=False
# )

# pages = loader.load()
# print("Loader Data  Successfully")



# # Split and embed documents
# r_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
# doc_list = [
#     Document(page_content=pg, metadata={"page_no": i + 1})
#     for i, page in enumerate(pages)
#     for pg in r_splitter.split_text(page.page_content)
# ]

# # Set up the embedding model and Qdrant vector store
# embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# qdrant_url = "https://bfdbdbbd-6a7e-48a7-83dc-8725edd588d3.us-east4-0.gcp.cloud.qdrant.io:6333"
# qdrant_key = "0ETKHECLieM7k-umC6Zec_eY1iFWAbugNDRBd76Ly-goAWfr54_6RA"
# collection_name = "Mishak_Lawbot_Baldwin"


# qdrant = QdrantVectorStore.from_documents(
#     doc_list,
#     embed_model,
#     url=qdrant_url,
#     prefer_grpc=True,
#     api_key=qdrant_key,
#     collection_name=collection_name,
# )

# # Initialize the language model
# openai_api_key = "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"
# llm_name = "gpt-4o-mini"
# llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai_api_key, temperature=0)


# prompt_template = """
# You are a professional legal assistant. Provide a clear, concise, and accurate response to the user's question based on the context provided. 

# Context:
# {context}

# Question:
# {question}

# Answer should be to the point and summarized.
# """




# # Define routes
# @app.route('/')
# def index():
#     return '<h1>Welcome to Matthew Legal Assistant</h1>'

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         # Get file from the request
#         file = request.files.get('file')
#         if not file:
#             return jsonify({"error": "No file provided"}), 400

#         # Save file temporarily
#         temp_dir = tempfile.mkdtemp()
#         temp_file_path = os.path.join(temp_dir, file.filename)
#         file.save(temp_file_path)

#         # Load and extract text from the document
#         loader = PyPDFLoader(temp_file_path, extract_images=True)
#         pages = loader.load()
        
#         # Combine all text from the document
#         document_text = "\n".join([page.page_content for page in pages])

#         # Create prompt - Default prompt to summarize the document
#         prompt = f"Please summarize the following document:\n{document_text}"

#         # Send the prompt to GPT
#         response = llm(prompt)
#         response_text = response.content if hasattr(response, "content") else str(response)

#         return jsonify({"response": response_text.strip()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/ask", methods=["POST"])
# def ask():
#     user_data = request.get_json()
#     query = user_data.get("question")

#     if not query:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         # Retrieve relevant documents from Qdrant
#         relevant_docs = qdrant.similarity_search(query, limit=3)
#         if not relevant_docs:
#             print("No relevant documents found.")
#             return jsonify({"response": "No relevant information found in the document."})

#         # Prepare context and format prompt
#         context = "\n\n".join(
#             [f"Page {doc.metadata['page_no']}:\n{doc.page_content}" for doc in relevant_docs]
#         )
#         formatted_prompt = prompt_template.format(question=query, context=context)
#         print("Formatted Prompt:", formatted_prompt)  # Debug the prompt content

#         # Get response from the language model
#         response = llm(formatted_prompt)
#         response_text = response.content if hasattr(response, "content") else str(response)

#         if not response_text:
#             print("No response from the language model.")
#             return jsonify({"response": "No answer could be generated for your question."})

#         # Include additional details in the response JSON
#         return jsonify(
#             {
#                 "response": response_text.strip(),
#                 "retrieved_documents": [doc.page_content for doc in relevant_docs],
#                 "formatted_prompt": formatted_prompt,
#             }
#         )

#     except Exception as e:
#         print("Error occurred:", str(e))
#         return jsonify({"error": "An error occurred while processing your request."}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
      

   
 