import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import logging
import pickle

logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) > 4 and block[4].strip():
                text += block[4] + "\n\n"
    doc.close()
    return text.strip()

pdf_path = "sample-service-manual 1.pdf"
full_text = extract_text_from_pdf(pdf_path)
print("Extracted text length:", len(full_text))

# Recursive Chunking
def recursive_chunking(text, size=500, overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    ).split_text(text)

chunks = recursive_chunking(full_text)
print("Total chunks:", len(chunks))

# Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Hybrid Retriever
def create_bm25(chunks):
    return BM25Retriever.from_texts(chunks, k=5)

def create_dense(chunks, embedding_model):
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore

bm25 = create_bm25(chunks)
dense = create_dense(chunks, embedding_model)
hybrid = EnsembleRetriever(
    retrievers=[bm25, dense.as_retriever(search_kwargs={"k": 5})],
    weights=[0.5, 0.5]
)

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

retriever = ContextualCompressionRetriever(
    base_retriever=hybrid,
    base_compressor=compressor
)

# Save everything needed
os.makedirs("vectorstore", exist_ok=True)
dense.save_local("vectorstore")

with open("vectorstore/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

# ─── MOST IMPORTANT ADDITION ───
with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
# ───────────────────────────────

print("Vector store + chunks saved. You can now run Flask app.")