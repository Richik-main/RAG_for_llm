import os
import uuid
import cloudpickle
import faiss
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS

class CustomMemoryDocStore:
    def __init__(self):
        self._dict = {}
        self._counter = 0
    
    def add_texts(self, texts, metadatas):
        for text, metadata in zip(texts, metadatas):
            doc_id = str(self._counter)
            self._dict[doc_id] = {"text": text, "metadata": metadata}
            self._counter += 1
    
    def get_text(self, doc_id):
        return self._dict[doc_id]["text"]
    
    def get_metadata(self, doc_id):
        return self._dict[doc_id]["metadata"]
    
    def save(self, path):
        with open(path, "wb") as f:
            cloudpickle.dump(self._dict, f)

    def load(self, path):
        with open(path, "rb") as f:
            self._dict = cloudpickle.load(f)
        self._counter = len(self._dict)

def load_vector_db(DB_PATH="./db/"):
    if os.path.exists(DB_PATH + "faiss.index"):
        print("Loading existing database...")
        index = faiss.read_index(DB_PATH + "faiss.index")
        docstore = CustomMemoryDocStore()
        docstore.load(DB_PATH + "memoryDocStoreDict.pkl")
        index_to_docstore_id_dict = cloudpickle.load(open(DB_PATH + "indexToDocStoreIdDict.pkl", "rb"))
        return FAISS(index, docstore, index_to_docstore_id_dict)
    else:
        print("Creating a new database...")
        index_to_docstore_id_dict = {}
        return FAISS(
            faiss.IndexFlatL2(768),
            CustomMemoryDocStore(),
            index_to_docstore_id_dict
        )

# Function to populate the vector database with documents.
# It processes each file in the 'wiki/' directory, splits the content into smaller chunks,
# and stores these chunks along with their metadata in the database.
def populate_vector_db(DB_PATH="./Users/richikghosh/Documents/Rag_project/db/"):
    db = load_vector_db(DB_PATH=DB_PATH)

    # Process each file in the 'wiki/' directory.
    for wiki_file in os.listdir("/Users/richikghosh/Documents/Rag_project/wiki/"):
        texts = []
        metadatas = []
        
        wiki_file_path  = os.path.join("/Users/richikghosh/Documents/Rag_project/wiki", wiki_file)
        wiki_chunks_dir = os.path.join("/Users/richikghosh/Documents/Rag_project/wiki_chunks", wiki_file)
        os.makedirs(wiki_chunks_dir, exist_ok=True)
       
        # Read the content of the file.
        with open(wiki_file_path, "r") as file:
            content = file.read()
        print(f"Processing file: {wiki_file_path}")
        # Split the content into smaller chunks for better manageability.
        text_splitter = TokenTextSplitter(chunk_size=256)
        chunks = text_splitter.split_text(content)
        print(f"Total chunks for {wiki_file}: {len(chunks)}")
        
        for chunk in chunks:
            random_uuid = str(uuid.uuid4())
            texts.append(chunk)
            
            wiki_chunk_file_path = os.path.join(wiki_chunks_dir, f"{random_uuid}.txt")
            with open(wiki_chunk_file_path, "w") as chunk_file:
                chunk_file.write(chunk)
            metadatas.append({
                'wiki_file_path': wiki_file_path,
                'wiki_chunk_file_path': wiki_chunk_file_path
            })

        # Add the text chunks and their metadata to the database.
        print(f"Adding {len(texts)} chunks to the database for file: {wiki_file_path}")
        db.add_texts(texts, metadatas)
        
    # Save the components of the database if the directory does not exist.
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    
    print("Saving database components...")
    db.docstore.save(DB_PATH + "memoryDocStoreDict.pkl")
    cloudpickle.dump(db.index_to_docstore_id, open(os.path.join(DB_PATH, "indexToDocStoreIdDict.pkl"), "wb"))
    faiss.write_index(db.index, os.path.join(DB_PATH, "faiss.index"))

    print("Database populated and saved successfully.")
    
    return db

# Ensure the 'wiki_chunks' directory exists
os.makedirs("/Users/richikghosh/Documents/Rag_project/wiki_chunks", exist_ok=True)

# Call the function to populate the vector database
populate_vector_db()
