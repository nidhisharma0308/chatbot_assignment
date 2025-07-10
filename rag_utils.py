import os
import re
from typing import List
import faiss

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from sentence_splitter import SentenceSplitter
from pypdf import PdfReader  


def extract_text_from_pdf(pdf_path: str) -> str:
	"""Extract raw text from a PDF file using pypdf"""
	reader = PdfReader(pdf_path)
	raw_text = ""
	for page in reader.pages:
		raw_text += page.extract_text() + "\n"
	return raw_text


def clean_text(raw_text: str) -> str:
	"""Clean whitespace and newlines from text"""
	text = re.sub(r"\s+", " ", raw_text.strip())
	return text


def chunk_text(text: str, chunk_size: int = 250) -> List[str]:
	"""Split cleaned text into sentence-aware chunks"""
	splitter = SentenceSplitter(language='en')
	sentences = splitter.split(text)
	chunks = []
	current = ""
	for sent in sentences:
		if len(current) + len(sent) < chunk_size:
			current += " " + sent
		else:
			chunks.append(current.strip())
			current = sent
	if current:
		chunks.append(current.strip())
	return chunks


import json

def load_and_process(file_path: str, save_chunks: bool = True, chunk_save_path: str = "chunks/chunks.json") -> List[Document]:
    """Extract, clean, chunk text and return LangChain Document objects.
       Optionally saves to a structured JSON file."""
    text = extract_text_from_pdf(file_path)
    text = clean_text(text)
    chunks = chunk_text(text)

    documents = []
    chunk_records = []

    for i, chunk in enumerate(chunks):
        documents.append(Document(page_content=chunk, metadata={"chunk_id": i}))
        chunk_records.append({
            "chunk_id": i,
            "text": chunk,
            "start_page": None,
            "source": os.path.basename(file_path)
        })

    if save_chunks:
        os.makedirs(os.path.dirname(chunk_save_path), exist_ok=True)
        with open(chunk_save_path, "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, indent=2, ensure_ascii=False)

    return documents


def create_or_load_vectorstore(docs: List[Document], db_path: str) -> FAISS:
	"""Build or load FAISS vector DB from documents"""
	model_name = r"C:/Users/Nidhi.Sharma_1/Desktop/rag_ass_bot/models/all-MiniLM-L6-v2"
	embedder = HuggingFaceEmbeddings(model_name=model_name)

	if os.path.exists(os.path.join(db_path, "index.faiss")):
		db = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
	else:
		db = FAISS.from_documents(docs, embedder)
		db.save_local(db_path)
	return db

def format_prompt(context: str, query: str) -> str:
    template = f"""
You are Algo, a strict AI assistant trained to answer questions **only** based on the contents of a given PDF document.

📌 Rules you must follow:
1. You must use **only the retrieved context** to answer the question.
2. If the question is **not clearly answerable from the context**, respond with exactly:
   👉 "NOT RELATED TO THE PDF."
3. Do **not** use outside knowledge or assumptions.
4. Do **not** attempt to explain topics not covered in the document.
5. Keep your answers **concise, factual, and context-specific**.

---

🧠 User Question:
{query}

📄 Retrieved Context (from PDF):
{context}

✍️ Your Response (answer or reject):
"""
    return template





