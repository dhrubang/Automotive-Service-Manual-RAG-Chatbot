import os
import pickle
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate
from groq import Groq

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key_change_me_in_production")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")

# ────────────────────────────────────────────────────────────────
#   Rebuild advanced retriever on startup (same as notebook)
# ────────────────────────────────────────────────────────────────

try:
    # Load chunks
    with open("vectorstore/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Same as creation script
    bm25 = BM25Retriever.from_texts(chunks, k=5)

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    dense = vectorstore.as_retriever(search_kwargs={"k": 5})

    hybrid = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5]
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

    retriever = ContextualCompressionRetriever(
        base_retriever=hybrid,
        base_compressor=compressor
    )

    print("Advanced hybrid + reranked retriever successfully rebuilt.")

except Exception as e:
    print("Failed to load/rebuild retriever:", str(e))
    raise

# ────────────────────────────────────────────────────────────────
#   Groq + Chain
# ────────────────────────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

prompt = PromptTemplate.from_template("""
You are an expert automotive technician extracting ONLY real specifications from the service manual context.
Extract NOTHING that is not explicitly stated.
Do NOT guess, infer, hallucinate, or add any extra text.

User query: {query}

Most relevant manual text (ranked by relevance):
{context}

Return ONLY a valid JSON array where the response will be easier to understand in key : value form where left side attribute tells about the the right side attribute

If no specification matches, return empty array: []
""")

def groq_llm(prompt_input) -> str:
    prompt_text = prompt_input.to_string() if hasattr(prompt_input, "to_string") else str(prompt_input)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

llm = RunnableLambda(groq_llm)

rag_chain = (
    {
        "query": RunnablePassthrough(),
        "context": lambda q: retriever.invoke(q)
    }
    | RunnablePassthrough.assign(
        context=lambda inputs: "\n\n".join(doc.page_content for doc in inputs["context"])
    )
    | prompt
    | llm
)

# ────────────────────────────────────────────────────────────────
#   Flask routes
# ────────────────────────────────────────────────────────────────

def ensure_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []

@app.route("/")
def index():
    ensure_chat_history()
    return render_template("index.html", chat_history=session['chat_history'])

@app.route("/chat", methods=["POST"])
def chat():
    ensure_chat_history()

    query = request.form.get("query", "").strip()
    if not query:
        return jsonify({"response": "[Please type a question]"})

    try:
        output = rag_chain.invoke(query)
    except Exception as e:
        output = f"[RAG error: {str(e)}]"

    session['chat_history'].append({"role": "user", "content": query})
    session['chat_history'].append({"role": "assistant", "content": output})

    return jsonify({"response": output})

@app.route("/new_chat", methods=["POST"])
def new_chat():
    session.pop('chat_history', None)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)