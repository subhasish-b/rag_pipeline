# RAG Pipeline Implementation
## Introduction: 
A RAG pipeline implementation from scratch*. The solution is a question answering Bot which provides answers to questions from a closed domain that the LLM is not aware of. In this case I have taken 2 chapters from Book in PDF format. The implementation is from End-to-End, reading from unstructured data (pdf), converting to structured data and storing in db, performing semantic search, implementing LLM, wrapping the code in an API and viewing results in streamlit UI.
*used low level libraries.

### Installation:
##### Pre-requisite:
•	Python 3.11
•	List of packages present in requirements.txt file installed (preferably in a virtual environment)
•	CUDA (if using a GPU, version to matched with GPU version) (optional)
•	Download HuggingFace models for faster implementation (optional)
### Implementation:
1.	Run “rag_pipeline.ipnyb”: This is very basic implementation from scratch without using any high level library. Sentence Embedding and similarity search are performed by sentence_transformer library. No database being used to save embeddings. All operations in-memory.
2.	Run “rag_pipeline.ipnyb”: A advance version of the previous file using ChromaDB as in-memory vector database capable of storing embedding with docs and metadata and performing question answering based on sentence similarity.
3.	Run “rag_pipeline_demo”: Few lines of codes that triggers the retrieval and rag files to produce answers. Best suitable for testing the solution in least amount of time.
4.	API run: The code is wrapped by fastapi. Running command “fastapi dev main.py –reload” will start the api. The api can be tested in the swagger page of fastapi.
5.	Streamlit run: Once the API is up, the streamlit app can hit the API and fetch answers. A very basic UI app. Command “streamlit run streamlit_app.py”.
6.	The config.py has all configuration parameters which can be changed before running the pipeline.


### Understanding the Flow: 
1.	The pdf is read, pre-processed, and converted to a python dictionary that contains list of sentences along with metadata such as page number, word count/page, sentence count/page, tokens/page etc.
2.	The list of sentences is further converted to chunk of sentences (optimized to token counts that can be handled by the embedding model and LLM without information loss). 
3.	The Chunk of sentences are converted embeddings. Two methods I have tried:
a.	Using sentence_transformer library. Only used in Jupyter notebook.
b.	Using ChromaDB library. Used in the API solution.
4.	Retrieval Phase: Using sentence similarity (either dot or cosine), fetched most similar docs (top_n) from the embeddings for the given query.
5.	Augmentation Phase: Augmented the retrieved passages to the LLM model Prompt. Now the LLM will be aware of the most similar passages of the given query.
6.	Generation Phase: Prompted LLM model to generate answer for the query based on similar context/passages provided.
   
#### Few Points to Note:
1.	The LLM model chosen is a very lightweight model to run in local systems (1.5B params). Accuracy of the answer can be improved by using a commercial grade robust LLM (70+B params).
2.	The retrieval model’s accuracy seems to be low, when queries are more reasoning based.
3.	The RAG pipeline won’t work in case of queries based on images in the book.
   
#### Known Bugs and Future Tasks:
1.	The Flash Attention module couldn’t be installed in windows – Open Bug in their discussion page. Running LLM in local will be slower.
2.	Couldn’t implement the code in Colab. Facing different bugs and due to time constraints, couldn’t error handle it. Working on it still.
3.	Dockerization in progress. 
