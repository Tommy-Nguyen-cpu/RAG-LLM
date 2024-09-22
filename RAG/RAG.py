from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from wikipediaapi import Wikipedia
import numpy as np

class RAG_Model():
    def __init__(self, embedding_model_name = "BAAI/bge-small-en-v1.5", llm_model_name = "Qwen/Qwen2-1.5B"):
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map = "auto") # Load LLM model and let the model detect with device to load model into.
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast = True) # Load tokenizer. Load rust-based fast variant of model if available.
        self.embedding = SentenceTransformer(embedding_model_name)
        self.wiki = Wikipedia("RAGBot/0.0", "en") # Initialize wiki object to grab info from wiki pages.

    def grab_docs(self, query, wiki_page_name, top_k = 5, num_overlapping_words = 10):
        paragraphs = self.process_wikipage(wiki_page_name, num_overlapping_words) # Grab content from wiki and process input strings.

        query_embeddings = self.embedding.encode(query, normalize_embeddings = True) # Encode user input query and normalize so we can use dot to find similarities.
        context_embeddings = self.embedding.encode(paragraphs, normalize_embeddings = True) # Encode context and normalize it so we can use dot product to find similarities.

        similarities = np.dot(context_embeddings, query_embeddings.T) # Calculate similarities between context and query.
        top_k_results = np.argsort(similarities)[::-1][:top_k] # Sort in descending order and grab top_k results.
        return np.array(paragraphs)[top_k_results], top_k_results

    def process_wikipage(self, wiki_page_name, num_overlapping_words):
        page =  self.wiki.page(wiki_page_name)# Grab wikipedia page text.
        paragraphs = page.text.split("\n\n") # Split wiki page into paragraphs.

        # Incorporate overlapping between text chunks.
        for i in range(1, len(paragraphs)):
            prev_paragraph = paragraphs[i-1].split() # Splits previous paragraph into list of words.
            last_overlap_words = np.array(prev_paragraph)[len(prev_paragraph) - num_overlapping_words::] # Grab last num words in para.
            paragraphs[i] = " ".join(last_overlap_words) + " " + paragraphs[i]
        
        return paragraphs