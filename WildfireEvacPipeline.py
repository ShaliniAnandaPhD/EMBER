import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from typing import List, Dict
import logging

class WildfireEvacPipeline:
    def __init__(self, retrieval_model: str, generation_model: str, index_name: str, es_host: str, es_port: int):
        """
        Initialize the WildfireEvacPipeline.

        Args:
            retrieval_model (str): Name or path of the pre-trained sentence-transformer model for retrieval.
            generation_model (str): Name or path of the pre-trained seq2seq model for generation.
            index_name (str): Name of the Elasticsearch index containing the documents.
            es_host (str): Elasticsearch host.
            es_port (int): Elasticsearch port.
        """
        # Initialize the retrieval model (sentence-transformer) for encoding queries and documents
        self.retrieval_model = SentenceTransformer(retrieval_model)
        
        # Initialize the tokenizer and generation model (seq2seq) for generating instructions
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model)
        self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model)
        
        # Initialize the Elasticsearch client for retrieving documents
        self.es = Elasticsearch(hosts=[{"host": es_host, "port": es_port}])
        
        # Set the Elasticsearch index name
        self.index_name = index_name
        
        # Initialize the generation pipeline with the tokenizer and model
        self.generation_pipeline = pipeline("text2text-generation", model=self.generation_model, tokenizer=self.tokenizer)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the top-k most relevant documents for a given query.

        Args:
            query (str): User query.
            top_k (int): Number of documents to retrieve.

        Returns:
            List[str]: List of retrieved document texts.
        """
        # Encode the query using the retrieval model
        query_vector = self.retrieval_model.encode(query)

        # Construct the Elasticsearch query
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                    "params": {"query_vector": query_vector.tolist()}
                }
            }
        }

        # Execute the Elasticsearch search
        response = self.es.search(index=self.index_name, body={"query": script_query, "size": top_k})

        # Extract the document texts from the search response
        retrieved_docs = [hit["_source"]["text"] for hit in response["hits"]["hits"]]

        return retrieved_docs

    def generate_instructions(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Generate instructions based on the user query and retrieved documents.

        Args:
            query (str): User query.
            retrieved_docs (List[str]): List of retrieved document texts.

        Returns:
            str: Generated instructions.
        """
        # Concatenate the retrieved documents
        context = " ".join(retrieved_docs)

        # Prepare the input for generation
        input_text = f"Query: {query}\nContext: {context}\nInstructions:"

        # Generate the instructions using the generation pipeline
        output = self.generation_pipeline(input_text, max_length=200, num_return_sequences=1)

        # Extract the generated instructions from the output
        instructions = output[0]["generated_text"]

        return instructions

    def run(self, query: str, top_k: int = 5) -> str:
        """
        Run the WildfireEvacPipeline for a given user query.

        Args:
            query (str): User query.
            top_k (int): Number of documents to retrieve.

        Returns:
            str: Generated instructions.
        """
        # Log the user query
        logging.info(f"User Query: {query}")

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        # Log the retrieved document IDs
        document_ids = [hit["_id"] for hit in self.es.search(index=self.index_name, body={"query": {"ids": {"values": retrieved_docs}}})["hits"]["hits"]]
        logging.info(f"Retrieved Document IDs: {document_ids}")

        # Generate instructions
        instructions = self.generate_instructions(query, retrieved_docs)

        # Log the generated instructions
        logging.info(f"Generated Instructions: {instructions}")

        return instructions