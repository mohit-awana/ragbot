from flask import Flask, render_template, request, Response
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer
from main import PredictionPipeline
import os
import faiss
import tqdm
import numpy as np



# Create a Flask web application
app = Flask(__name__)

# Initialize a PredictionPipeline object
pipeline = PredictionPipeline()


# Load the dataset (train split) that's already chunked
print(f"{Fore.RED}1.) Loading and chunking dataset...{Style.RESET_ALL}")

global documents 
documents = pipeline.datasets()
# Generate embeddings for the documents using SentenceBERT and index them using FAISS



print(f"{Fore.RED}2.) Generating Embedding Vectors using Sentence BERT and indexing using FAISS...{Style.RESET_ALL}")

sentence_bert = SentenceTransformer('all-mpnet-base-v2') # all-mpnet-base-v2

if not os.path.exists(".indices/index_latest.idx"):
    os.mkdir(".indices/index_latest.idx")
    index = faiss.IndexFlatL2(768)
    batch_size = int(os.getenv("BATCH_SIZE"))
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding Documents", colour="green"):
        batch = documents[i:i+batch_size]
        embeds = sentence_bert.encode(batch)
        # to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
        # index.add(np.array(to_upsert))
        index.add(np.array(embeds))
        # Save the index
        faiss.write_index(index, ".indices/index_latest.idx")
else:
    # Load the index
    indexX = faiss.read_index(".indices/index_latest.idx")

# Define a route for the root URL ('/'), rendering an HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Retrieve the top-k documents for a query using the FAISS index
print(f"{Fore.RED}3.) Retrieve Top-K documents using FAISS...{Style.RESET_ALL}")

#retriever = pipeline.search()

# Define a route for the '/stream' URL
@app.route('/stream')
def stream():
    # Retrieve input text from query parameters
    query = request.args.get('text')
    #print(query)
    #print("----------------------xxxx---------")
    #print(indexX.search(sentence_bert.encode(query), 20))
    docs = pipeline.search(documents, sentence_bert, indexX, query, 20)


    # Rerank the top-n documents using DistilBERT
    print(f"{Fore.RED}4.) Re-Ranking documents using distilBERT and retrieving Top-N documents...{Style.RESET_ALL}")
    reranked_docs = pipeline.rerank(docs, query, top_n=5)
    context = "\n".join([doc[0] for doc in reranked_docs])

    print(f"{Fore.RED}5.) Generate reponse using LLM...{Style.RESET_ALL}")
    return Response(pipeline.make_predictions(query, context), content_type='text/event-stream')


if __name__ == '__main__':
    # Listen on all available network interfaces ('0.0.0.0') on port 8000
    app.run(host='0.0.0.0', port=8000)