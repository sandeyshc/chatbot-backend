from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import traceback
import requests
from bs4 import BeautifulSoup
# from netlify import handler
from mangum import Mangum

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
content_store = []

def read_pdf(file_path: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        length_function=len
    )
    plan_name = os.path.basename(file_path).split('.')[0]
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text().replace("\n", " ")
        chunks = text_splitter.split_text(text)
        table_chunks = [f"IF USER ASKS DETAILS ABOUT THIS PLAN then Read below and answer. PLAN: {plan_name}\n\n{chunk}" for chunk in chunks]
        return table_chunks

def read_docx(file_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        length_function=len
    )
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        para_text = para.text.strip()
        if para_text:
            text += para_text + " "
    chunks = text_splitter.split_text(text)
    return chunks

def scrape_website(base_url: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        length_function=len
    )
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to retrieve data from {base_url}")
        return []
    soup = BeautifulSoup(response.content, 'html.parser')
    support_links = soup.find_all('a', href=True)
    support_pages = [link['href'] for link in support_links if 'support' in link['href']]
    all_content = []
    for page in support_pages:
        page_url = page if page.startswith("http") else base_url + page
        page_response = requests.get(page_url)
        if page_response.status_code == 200:
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            page_content = page_soup.get_text()
            split_texts = text_splitter.split_text(page_content)
            all_content.extend(split_texts)
        else:
            print(f"Failed to retrieve data from {page_url}")
    return all_content

def preprocess_documents():
    global content_store
    pdf_paths = [
        "America's_Choice_2500_Gold_SOB (1) (1).pdf",
        "America's_Choice_5000_Bronze_SOB (2).pdf",
        "America's_Choice_5000_HSA_SOB (2).pdf",
        "America's_Choice_7350_Copper_SOB (1) (1).pdf"
    ]
    docx_paths = ["America's_Choice_Medical_Questions_-_Modified_(3) (1).docx"]
    website_urls = ["https://www.angelone.in/support"]
    for pdf_path in pdf_paths:
        content_store += read_pdf(pdf_path)
    for docx_path in docx_paths:
        content_store += read_docx(docx_path)
    for url in website_urls:
        content_store += scrape_website(url)

def identify_context(question, threshold=0.5):
    sbert_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    question_embedding = sbert_model.encode(question, convert_to_tensor=True)
    best_match = None
    best_score = -1
    for content in content_store:
        content_embedding = sbert_model.encode(content, convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_embedding, content_embedding).max().item()
        if score > best_score:
            best_score = score
            best_match = content
    return best_match if best_score >= threshold else None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    preprocess_documents()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(request: dict):
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        context = identify_context(query)
        if not context:
            return {"response": "I don't know"}
        result = qa_pipeline(question=query, context=context)
        return {"response": result['answer']}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# handler = handler(app)

handler = Mangum(app)