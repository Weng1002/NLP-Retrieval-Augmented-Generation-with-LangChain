import os
import json
import bs4
import nltk
import torch
import pickle
import numpy as np

# from pyserini.index import IndexWriter
# from pyserini.search import SimpleSearcher
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Hugging Face Login (Optional if token is already set or not pushing)
# from huggingface_hub import login
# hf_token = "YOUR_HF_TOKEN_HERE"
# login(token=hf_token, add_to_git_credential=True)

# TODO1: Set up the environment of Ollama
# Note: In a script, we assume Ollama is already installed and running.
# !pip install colab-xterm
# %load_ext colabxterm
# !curl -fsSL https://ollama.com/install.sh | sh
# %xterm
# ollama serve &
# ollama pull llama3.2:1b

# Setting up the model
MODEL = "llama3.2:1b" 
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"

# Initialize Ollama
print("Initializing Ollama...")
llm = Ollama(model=MODEL)
# Test Ollama
# response = llm.invoke("What is the capital of Taiwan?")
# print(response)

# TODO2: Load the cat-facts dataset and prepare the retrieval database
print("Loading dataset...")
if not os.path.exists('./datasets/cat-facts.txt'):
    # Download if not exists (using subprocess or just warning)
    print("Downloading cat-facts.txt...")
    os.system("wget https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt -O ./datasets/cat-facts.txt")

# TODO2-1: Load the cat-facts dataset
with open('./datasets/cat-facts.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# Grouped Chunking (5 lines per document) - Optimization from Q3
chunk_size = 5
refs = []
for i in range(0, len(lines), chunk_size):
    chunk = " ".join(lines[i:i+chunk_size])
    refs.append(chunk)

print(f'Loaded {len(refs)} grouped documents (from {len(lines)} facts).')

docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(refs)]

# Create embedding model
print("Initializing Embedding Model...")
model_kwargs = {'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# TODO2-2: Prepare the retrieval database
print("Creating Vector Store...")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    collection_name="cat_facts"
)

# TODO5 Optimization: Reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

print("Initializing Reranker...")
# 1. Setup Cross-Encoder for Reranking
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# 2. Setup Reranker (Top 3)
compressor = CrossEncoderReranker(model=model, top_n=3)

# 3. Setup Base Retriever (Fetch 20 candidates)
base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20} 
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

# TODO3: Set up the `system_prompt` and configure the prompt.
# Few-Shot Prompt - Optimization from Q2
system_prompt = (
    "You are a helpful assistant. "
    "Answer the question based ONLY on the following context. "
    "Keep your answer concise and to the point. "
    "\n\n"
    "Examples:"
    "\nQ: How much of a day do cats spend sleeping?"
    "\nA: Two thirds"
    "\nQ: What is a group of cats called?"
    "\nA: Clowder"
    "\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# TODO4: Build and run the RAG system
# TODO4-1: Load the QA chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# TODO4-2: Create retrieval chain
# Use the compression_retriever instead of the basic retriever
chain = create_retrieval_chain(compression_retriever, question_answer_chain)

# Load queries and answers
queries = []
answers = []

if not os.path.exists('datasets/questions_answers.txt'):
    print("Error: datasets/questions_answers.txt not found.")
else:
    with open('datasets/questions_answers.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                queries.append(lines[i])
                answers.append(lines[i+1])

    print(f"Loaded {len(queries)} queries and answers.")

    results = []
    correct_count = 0
    recall_1_count = 0
    recall_5_count = 0 

    print(f"Start evaluating {len(queries)} questions...")

    for i, query in tqdm(enumerate(queries), total=len(queries)):
        # TODO4-3: Run the RAG system
        response = chain.invoke({"input": query})
        
        generated_answer = response["answer"]
        ground_truth = answers[i]
        retrieved_docs = response["context"]
        
        # Evaluation 1: Exact Match (EM)
        is_correct = ground_truth.lower() in generated_answer.lower()
        if is_correct:
            correct_count += 1
            
        # Evaluation 2: Recall 
        # Recall@1
        if len(retrieved_docs) > 0 and ground_truth.lower() in retrieved_docs[0].page_content.lower():
            recall_1_count += 1
            
        # Recall@5
        found_in_docs = False
        for doc in retrieved_docs[:5]:
            if ground_truth.lower() in doc.page_content.lower():
                found_in_docs = True
                break
        if found_in_docs:
            recall_5_count += 1
            
        results.append({
            "Query": query,
            "Ground_Truth": ground_truth,
            "Prediction": generated_answer
        })

    with open('rag_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Exact Match (EM): {correct_count/len(queries):.2%}")
    print(f"Recall@1: {recall_1_count/len(queries):.2%}")
    print(f"Recall@5: {recall_5_count/len(queries):.2%}")


# ================= TODO5 Experiment: Counterfactual Analysis =================
# This section runs the counterfactual experiment for analysis
print("\nRunning Counterfactual Injection Experiment (Analysis Q5)...\n")

counterfactual_tests = [
    {
        "query": "How much of a day do cats spend sleeping?",
        "ground_truth": "Two thirds",
        "fake_context": "Scientific research has proven that cats never sleep. They are awake 24 hours a day watching us."
    },
    {
        "query": "What is a group of cats called?",
        "ground_truth": "Clowder",
        "fake_context": "A group of cats is officially called a 'Gaggle' in modern terminology."
    },
    {
        "query": "Why don't cats have a sweet tooth?",
        "ground_truth": "Taste mutation",
        "fake_context": "Cats actually love sweets! They have a highly developed sweet tooth and crave sugar all the time."
    }
]

for test in counterfactual_tests:
    print(f"Q: {test['query']}")
    print(f"Ground Truth: {test['ground_truth']}")
    print(f"Injected Counterfactual: {test['fake_context']}")
    
    # Inject fake context directly into the QA chain
    response = question_answer_chain.invoke({
        "input": test['query'],
        "context": [Document(page_content=test['fake_context'])]
    })
    
    print(f"Model Prediction: {response}")
    print("-" * 50)
