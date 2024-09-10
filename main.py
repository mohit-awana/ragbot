#from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import requests, re, urllib.parse
from docx import Document




class PredictionPipeline:
    def __init__(self):
        self.model_name = 'openai-community/gpt2'##'EleutherAI/gpt-neo-125m'#'EleutherAI/gpt-neo-2.7B'#'openai-community/gpt2' #'TinyLlama/TinyLlama-1.1B-step-50K-105b' 
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=2048)
        self.temperature = 0.7
        self.sentence_transformer_modelname = 'sentence-transformers/all-mpnet-base-v2' # 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_paths = ['criminal_law.docx','doc1.docx','doc2.docx']
        self.rerank_model = 'sentence-transformers/msmarco-distilbert-base-v3'
        print(f"1. Device being utilized: {self.device} !!!")
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def datasets(self):

 
        full_text = []

        for file_path in self.file_paths:
            doc = Document(file_path)
            current_page_text = []

            for para in doc.paragraphs:
                # Accumulate text until a page break is encountered
                current_page_text.append(para.text)

                # Check for a page break (handled as a hard break in Word)
                if para.text == "":
                    page_text = " ".join(current_page_text).strip()
                    # Only add text from pages with 50 or more words
                    if len(page_text.split()) >= 50:
                        full_text.append(page_text)
                    current_page_text = []  # Reset for the next page

            # Add the last page if it exists and passes the word count check
            if current_page_text:
                page_text = " ".join(current_page_text).strip()
                if len(page_text.split()) >= 50:
                    full_text.append(page_text)
        
        text = " ".join(full_text)


        # # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        texts = text_splitter.split_text(text)
        # split into chunks
        # text_splitter = SentenceTransformersTokenTextSplitter(
        # chunk_size=1000,
        # chunk_overlap=200
        # ,model_name = 'sentence-transformers/all-mpnet-base-v2'
        # ,tokens_per_chunk = 10
        # )

        # texts = text_splitter.split_text(text)

        return texts
    
    # def get_embedding(self, text) -> list:
    #     model = SentenceTransformer(self.sentence_transformer_modelname)  # all-mpnet-base-v2
    #     embeddings = model.encode(text)
    #     return embeddings
    
    def search(self, documents, embed_model, index, query, top_k) -> list:
        query_embedding = embed_model.encode(query).reshape(1, -1)

        distances, indices = index.search(query_embedding, top_k)
        ret_doc = [documents[int(idx)] for idx in indices[0]]
        return ret_doc

    def rerank(self, documents, query, top_n=10) -> list:
        model = SentenceTransformer(self.rerank_model)
        query_embedding = model.encode(query, convert_to_tensor=True)
        document_embeddings = model.encode(documents, convert_to_tensor=True)
        
        # Compute cosine similarities between the query and all documents
        similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

        # Combine the scores with the documents
        doc_scores = zip(documents, similarities)

        # Sort documents by their scores in descending order
        reranked_documents = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return reranked_documents[:top_n]
    


    

    def make_predictions(self, question, context, top_n_values=10):

        context_prompt = """
Please use the following context to answer the question. Provide a clear and concise answer solely based on the information provided in context.   
Context:
    """
    
        Answer_prompt = """
Question: """ 


        # contexts =  [context[i:i+500] for i in range(0, len(context), 500)]
        # for chunk in contexts:
        #     summary = f"{context_prompt}\n{chunk}\n\n{Answer_prompt}\n{question}"
        summary = f"{context_prompt}\n{context}\n\n{Answer_prompt}\n{question}"

        print("Generated Prompt:\n", summary)
        # Define stop words or phrases that should stop the generation
        
        stop_words = ["."]

        def detect_stop_word(output, stop_words):
            """ Check if any stop word is in the generated output. """
            for stop_word in stop_words:
                if stop_word in output:
                    return True
            return False

        #max_context_len = 1024  # For example, you can limit it to a smaller size
        inputs = self.tokenizer(summary, padding='longest'  # Right padding for GPT-based models
            ,truncation=True,  return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
  
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=500
                        
                            ,do_sample=True
                            ,temperature=self.temperature
                            ,top_p=0.95,
                            top_k=50,
                            repetition_penalty=2.1, pad_token_id=self.tokenizer.eos_token_id)
        
        token_count = 0
        def generate_thread(model, **kwargs):
            # Perform text generation
            output = model.generate(**kwargs)
            #print("Generated output:", output)
            return output
             


        # thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        # Start the thread
        thread = Thread(target=generate_thread, args=(self.model,), kwargs=generation_kwargs)

        thread.start()

        
        for token in streamer:
            token_count += 1
            yield f"data: {token}\n\n"  # Format for SSE
            
            if token_count > 80 and detect_stop_word(token, stop_words):
                print("Stop word detected, terminating generation early.")
                break  # Terminate generation if a stop word is detected

        thread.join()
        yield "data: END\n\n"


