# Automotive Service Manual RAG Chatbot

## Project Overview

Welcome to the **Automotive Service Manual RAG Chatbot**! This project leverages state-of-the-art Retrieval-Augmented Generation (RAG) to create an intelligent assistant that extracts precise specifications, procedures, and insights from automotive service manuals.

The chatbot is powered by **Groq's Llama-3.3-70B** model for generation, **FAISS** for vector storage, **BM25** for sparse retrieval, and a **cross-encoder** for intelligent reranking. It's deployed as a Flask web application with a clean, intuitive UI—perfect in the form of a chatbot to get easy answers to all queries

---

## 1. Evaluating Chunking and Retrieval Strategies

To ensure high-quality retrieval, we first benchmarked various chunking and retrieval methods in the **`Evaluating_Chunking_and_Retrieval_Strategies_for_Service_Manual_RAG_Systems.ipynb`** notebook.

### Chunking Methods Compared:

- **Fixed-size chunking**: Simple, uniform chunks with consistent token counts
- **Recursive chunking**: Adaptive splitting that respects document structure (paragraphs, sections)
- **Semantic chunking**: Uses embeddings to split based on semantic meaning shifts

### Retrieval Methods Compared:

- **BM25**: Sparse, keyword-based retrieval (traditional IR)
- **Dense retrieval**: Vector similarity using HuggingFace's `all-MiniLM-L6-v2` embeddings
- **Hybrid retrieval**: Intelligent ensemble combining BM25 + Dense methods
- **Reranked Hybrid**: Adds a cross-encoder (`ms-marco-MiniLM-L-6-v2`) for relevance scoring

### Evaluation Metrics:

- **Recall@K**: How many relevant documents are retrieved in the top K results
- **MRR (Mean Reciprocal Rank)**: Position of the first relevant hit
- **Precision@K**: Relevance quality of top K results
- **NDCG@K**: Normalized discounted cumulative gain (ranking quality)
- **Latency**: Query response time

We tested against a synthetic dataset of **15 queries** with ground-truth passages from the manual.

### Key Findings:

- **Recursive chunking** excelled in preserving context (e.g., keeping tables and procedures intact)
- **Reranked Hybrid retrieval** topped all metrics:
  - Recall@5: **~0.54**
  - MRR: **~0.61**
  - Precision@5: **~0.39**
  - NDCG: **~0.70**
- Reranking latency (**~0.7s**) was acceptable for our offline use case

We selected **Recursive Chunking + Reranked Hybrid Retrieval** as the optimal configuration for th e RAG pipeline.

---

## 2. Building the RAG Pipeline

With the best strategy identified, we prototyped the complete RAG pipeline in **`Automotive_Service_Manual_RAG_Pipeline.ipynb`**:

### Pipeline Architecture:

1. **PDF Extraction**
   - Used **PyMuPDF (fitz)** to extract clean text blocks
   - Processed the **852-page** sample manual efficiently

2. **Chunking**
   - Applied recursive splitting with `chunk_size=500` and `overlap=50`
   - Generated **~2125 meaningful chunks** from the manual

3. **Embedding & Storage**
   - Embedded chunks with **HuggingFace's all-MiniLM-L6-v2**
   - Stored vectors in **FAISS** for fast dense search
   - Built parallel **BM25 index** for keyword-based retrieval

4. **Retrieval Chain**
   - Implemented **ensemble hybrid retriever** (BM25 + Dense)
   - Added **cross-encoder reranker** for top_n=5 results
   - Integrated via **LangChain** for modularity

5. **Generation**
   - Prompted **Groq's Llama-3.3-70B** with strict template:
     > *"Extract ONLY real specifications... Return ONLY a valid JSON array... If no match, return []"* (as in the assignment it was asked to return in **json or csv** only )

   - Enforces **zero hallucinations**

   - Outputs are structured key-value pairs: `e.g. [{"Brake caliper guide pin bolts": "37 Nm, 27 lb-ft"}]`

6. **Testing & Validation**
   - Ran sample queries and validated JSON outputs against manual

This notebook served as the blueprint, form which a chatbot was built

---

## 3. Creating the Vector Store

Before deployment, we automated vector store creation in **`create_vectorstore.py`**:

### What it does:

- Extracts text from `sample-service-manual 1.pdf`
- Applies recursive chunking strategy
- Builds **BM25**, **FAISS**, **hybrid**, and **reranked retrievers**
- Saves everything to `vectorstore/` directory:
  - FAISS index (vector database)
  - Pickled chunks (text segments)
  - BM25 retriever (keyword index)

### Usage:

Run this **once** or whenever the manual updates:

```bash
python create_vectorstore.py
```

The script is efficient and lightweight, completing in under a minute for most documents.

---

## 4. Developing the Chatbot

Finally, we wrapped everything in a user-friendly **Flask web application** (`app.py`):

### Backend Architecture:

- **Pre-loaded retriever**: Loads the pre-built vector store on startup for instant responses
- **LangChain integration**: Uses Runnable chains for `query → retrieve → prompt → LLM`
- **Session management**: Flask sessions store chat history for multi-turn conversations
- **Error handling**: Clean fallbacks for edge cases (e.g., `[RAG error]`)

### Frontend Features:

- **Modern chat interface**: Clean HTML/CSS/JS with great chatbot interface
- **Responsive design**: Works seamlessly on desktop
- **Creative UI touches**:
  - Blue-themed color scheme mimicking professional chat apps
  - Rounded message bubbles with subtle shadows
  - Emoji-enhanced welcome message
  - "New Chat" button for easy conversation resets
- **JSON display**: Structured responses shown as-is for easy parsing


### Frontend Features:

- **Modern chat interface**: Clean HTML/CSS/JS with chat bubbles and message threading
- **Responsive design**: Works seamlessly on desktop and mobile
- **Creative UI touches**:
  - Blue-themed color scheme mimicking professional chat apps
  - Rounded message bubbles with subtle shadows
  - Emoji-enhanced welcome message
  - "New Chat" button for easy conversation resets
- **JSON display**: Structured responses shown as-is for easy parsing

### Project Structure:

```
.
├── app.py
│   # Main Flask application - handles routes, RAG pipeline orchestration, and Groq API calls
│
├── Automotive_Service_Manual_RAG_Pipeline.ipynb
│   # End-to-end RAG pipeline implementation notebook with PDF processing, chunking, 
│   # embedding generation, retrieval testing, and LLM integration experiments
│
├── create_vectorstore.py
│   # Script to process PDF, apply recursive chunking, generate embeddings using 
│   # all-MiniLM-L6-v2, and build FAISS index with BM25 hybrid retrieval
│
├── Evaluating_Chunking_and_Retrieval_Strategies_for_Service_Manual_RAG_Systems.ipynb
│   # Comprehensive notebook comparing chunking strategies (Fixed, Recursive, Semantic)
│   # and retrieval methods (BM25, Dense, Hybrid, Reranked) with performance metrics
│
├── memory.txt
│   # Stores session memory, conversation state, or runtime configuration data
│
├── requirements.txt
│   # Python dependencies required to run the project (LangChain, FAISS, Flask, etc.)
│
├── sample-service-manual 1.pdf
│   # Source automotive service manual (852 pages) used for RAG knowledge base
│
├── working_chatbot_screenshot1.png
│   # Screenshot demonstrating torque specification query result with JSON output
│
├── working_chatbot_screenshot2.png
│   # Screenshot showing pinpoint test and U-joint feature query with context-aware responses
│
├── static/
│   # Static frontend assets for the web interface
│   ├── css/
│   │   └── style.css
│   │       # Custom styling for the chatbot UI (blue theme, chat bubbles, responsive design)
│   └── js/
│       └── script.js
│           # Frontend interaction logic (message handling, API calls, UI updates)
│
├── templates/
│   └── index.html
│       # Main HTML template for the chatbot UI (chat interface, input form, message display)
│
├── vectorstore/
│   # Generated directory containing FAISS index, pickled chunks, and BM25 retriever data
│   # (created by running create_vectorstore.py)
│
└── venv/
    # Python virtual environment (not pushed to GitHub, created locally)
```

---

## Dependencies and Tech Stack

### Core Libraries:

- **LangChain** (0.1.20): Orchestration framework for RAG pipeline
- **FAISS**: Efficient vector similarity search
- **Sentence-Transformers**: Embedding generation
- **Groq SDK**: LLM API integration
- **Flask**: Web application framework
- **PyMuPDF (fitz)**: PDF text extraction
- **Rank-BM25**: Sparse retrieval

### Models:

- **all-MiniLM-L6-v2**: Fast, accurate embeddings (384 dimensions)
- **ms-marco-MiniLM-L-6-v2**: Cross-encoder for reranking
- **Llama-3.3-70B**: Powerful language model for generation (via Groq)

---

## 📸 Demo Screenshots

The demo screenshots of the chatbot are provided at : 

### Working Chatbot Screenshot 1

working_chatbot_screenshot1.png

### Working Chatbot Screenshot 2

working_chatbot_screenshot2.png

---

## Quick Start: How to Run the Chatbot

Follow these steps to get the chatbot up and running on your local machine.

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/dhrubang/Automotive-Service-Manual-RAG-Chatbot.git

# Navigate to the project directory
cd Automotive-Service-Manual-RAG-Chatbot
```

### 2. Set Up Environment Variables

> **Note:** The API key has been removed from the `.env` file. You can create your own API key from: https://console.groq.com/

Create a `.env` file in the root directory with the following content:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Create and Activate Virtual Environment

It's highly recommended to use a virtual environment to isolate dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

### 5. Build the Vector Store

This step processes the sample PDF (`sample-service-manual 1.pdf`) into chunks, embeds them, and creates a hybrid vector store:
```bash
python create_vectorstore.py
```

**Output:** This generates a `vectorstore/` directory containing the FAISS index and pickled chunks/BM25 retriever.

### 6. Launch the Chatbot

Start the Flask application:
```bash
python app.py
```

Open your browser and navigate to **http://127.0.0.1:5000**

### 7. Start Chatting! 💬

- Type queries like *"torque specification for brake caliper guide pin bolts"* or *"what is pinpoint test"*
- Use the **"+ New Chat"** button to reset the conversation history
- Press **Ctrl+C** in the terminal to stop the server

---


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Groq** for providing blazing-fast LLM inference
- **LangChain** for the excellent RAG framework
- **HuggingFace** for open-source embedding models

---
