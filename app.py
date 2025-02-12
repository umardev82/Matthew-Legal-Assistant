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
CORS(app, resources={r"/*": {"origins": "*"}})

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

@app.route('/welcome')
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
    app.run(host='0.0.0.0',port=8000)


