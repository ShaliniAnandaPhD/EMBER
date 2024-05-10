"""
WildfireRAG: A sophisticated Retrieval-Augmented Generation (RAG) system for generating wildfire evacuation instructions.

The system consists of the following components:
1. Document Retrieval: Retrieves relevant documents from an Elasticsearch index based on the user's query using cosine similarity search.
2. Instruction Generation: Generates instructions using a pre-trained GPT-2 model, taking the retrieved documents and user query as input.
3. User Query Processing: Processes the user's query, location, and preferences to generate personalized instructions.
4. FastAPI Integration: Exposes the functionality through a FastAPI web API endpoint.

Assumptions:
1. The user query is expected to contain a 'query' field (string), a 'location' field (string), and a 'preferences' field (dictionary).
2. An Elasticsearch index named "wildfire_documents" exists and contains the documents for retrieval, with each document having a "text" field.
3. Pre-trained models (GPT-2 and sentence transformer) are suitable for the task and have been trained on relevant data.
4. Additional processing based on user preferences may be required but is not implemented in this code.
5. Necessary dependencies are installed in the environment.
6. An Elasticsearch server is running and accessible at the default location (localhost:9200).
"""

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from fastapi import FastAPI
from pydantic import BaseModel

class UserQuery(BaseModel):
   query: str
   location: str
   preferences: dict

class WildfireRAG:
   def __init__(self):
       self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
       self.generator = GPT2LMHeadModel.from_pretrained('gpt2')
       self.retriever = SentenceTransformer('all-mpnet-base-v2')
       self.es = Elasticsearch()
       self.index_name = "wildfire_documents"

   def retrieve_documents(self, query, top_k=5):
       query_vector = self.retriever.encode([query])[0]
       script_query = {
           "script_score": {
               "query": {"match_all": {}},
               "script": {
                   "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                   "params": {"query_vector": query_vector.tolist()}
               }
           }
       }
       response = self.es.search(index=self.index_name, body={"query": script_query, "size": top_k})
       retrieved_docs = [hit['_source']['text'] for hit in response['hits']['hits']]
       return retrieved_docs

   def generate_instructions(self, retrieved_docs, user_query):
       prompt = f"User Query: {user_query}\nRetrieved Documents: {retrieved_docs}\nInstructions:"
       input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
       output = self.generator.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
       instructions = self.tokenizer.decode(output[0], skip_special_tokens=True)
       return instructions

   def process_query(self, user_query: UserQuery):
       query = user_query.query
       location = user_query.location
       preferences = user_query.preferences

       retrieved_docs = self.retrieve_documents(query)
       instructions = self.generate_instructions(retrieved_docs, query)

       # Assumption: Additional processing based on user preferences (not implemented)
       # ...

       return {"instructions": instructions}

app = FastAPI()
wildfire_rag = WildfireRAG()

@app.post("/generate_instructions")
async def generate_instructions(user_query: UserQuery):
   instructions = wildfire_rag.process_query(user_query)
   return instructions

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)