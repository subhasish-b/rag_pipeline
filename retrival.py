import pymupdf
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer
import torch
import numpy as np
from chromadb import PersistentClient, Client
from chromadb.utils import embedding_functions
from config import num_sentence_chunk_size, embed_model_name, collection_name, pdf_path, initial_page, final_page, top_n_results

class LoadAndRetrieve():

    def __init__(self):
        self.chroma_client = PersistentClient()
        self.huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model_name)


    def format_text(self, text):
        """
        Performs formatting operation on text, removes unncessary characters.

        Parameters:
            text (str): Text input to be formatted.

        Returns:
            str: A formatted string of the text provided.
        """
        clean_text = text.replace("\n", " ").strip()
        return clean_text
    
    def read_pdf(self):
        """
        Opens a PDF file, reads its text content page by page, and collects statistics.

        Parameters:
            pdf_path (str): The file path to the PDF document to be opened and read.
            initial_page (int): The first page number of the PDF to read
            final_page (int): The last page number of the PDF to read

        Returns:
            list[dict]: A list of dictionaries, each containing the page number,
            character count, word count, sentence count, token count, and the extracted text
            for each page.
        """
        doc = pymupdf.open(pdf_path)
        text_pages = []
        for page_number, page in enumerate(doc.pages(initial_page,final_page)):
            #Iterate over each page in the PDF
            #Gets the text in the page and stores various information as dict in a list
            text = page.get_text()
            text = self.format_text(text)
            text_pages.append({"page_number": page_number + 19,  # adjust page numbers since our PDF starts on page 42
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                    "text": text})
            
        return text_pages
    
    # Create a function that recursively splits a list into desired sizes
    def split_list(self, input_list, slice_size):
        """
        Splits the input_list into sublists of size slice_size.

        Parameters:
        input_list (list): Input list to slice into sublists
        slice_size (int): Input slicing size

        Return:
        list[list[str]]: List of subsets of the complete input list
        """
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    
    def pre_process(self, text_pages):
        """
        Converts string to sentences and then list of sentences to chunks of sentences

        Parameters:
        text_pages (dict): A dictionary containing list of strings

        Returns:
        dict: A dictionary containing chucks of sentences along with other important keys
        """
        nlp = English()
        # Add a sentencizer pipeline
        nlp.add_pipe("sentencizer")
        for item in text_pages:
            item["sentences"] = list(nlp(item["text"]).sents)
            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            # Count the sentences 
            item["page_sentence_count_spacy"] = len(item["sentences"])
            item["sentence_chunks"] = self.split_list(input_list=item["sentences"],
                                                slice_size=num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        return text_pages
    
    def create_embeddings(self):
        """
        Converts sentence to embedding and stores in chromadb
        """
        if collection_name in [c.name for c in self.chroma_client.list_collections()]:
            collection = self.chroma_client.get_collection(name=collection_name,embedding_function= self.huggingface_ef)
            if collection.count() != 0:
                print("collection_exist")
                return collection
            self.chroma_client.delete_collection(name=collection_name)
            
        pdf_pages = self.read_pdf()
        text_pages = self.pre_process(pdf_pages)
        documents = []
        metadata = []
        ids = []
        doc_id = 1
        for item in text_pages:
            for sentence_chunk in item["sentence_chunks"]:
                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                documents.append(joined_sentence_chunk)
                metadata.append({"page_no":item["page_number"],
                                "chunk_char_count":len(joined_sentence_chunk),
                                "chunk_word_count":len([word for word in joined_sentence_chunk.split(" ")]),
                                "chunk_token_count":len(joined_sentence_chunk) / 4
                                })
                ids.append(str(doc_id))
                doc_id += 1
        
        collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function= self.huggingface_ef)
        collection.add(
            documents=documents, # we embed for you, or bring your own
            metadatas=metadata, # filter on arbitrary metadata!
            ids=ids # must be unique for each doc 
            )
        
        return collection
        
    def query(self, collection, query):
        """
        Given a query, the method searches the relevant documents and return top n simillar documents.

        Parameters:
        collection (Chromadb.Collection): A database containing document embeddings along with metadata
        query (str): User query

        Returns:
        list[dict]: Return list  of top n results with other metadata
        """
        results = collection.query(
            query_texts=query,
            n_results=top_n_results)
        
        return_results = []
        for page, passage in zip(results["metadatas"][0], results["documents"][0]):
            query_results = {}
            query_results["page_no"]= page["page_no"]
            query_results["passage"]= passage
            return_results.append(query_results)
        
        return return_results


        

            



