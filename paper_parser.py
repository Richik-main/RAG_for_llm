
#%%

########################## PARSE PDF and form texts ########################## 
import PyPDF2
import pandas as pd
import os

def parse_pdf_to_dataframe(pdf_files):
    # List to store the data for each PDF
    data = []

    # Loop through each PDF file
    for pdf_file in pdf_files:
        # Open the PDF file in binary mode
        with open(pdf_file, 'rb') as file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = reader.metadata
            num_pages = len(reader.pages)
            
            # Extract content from each page
            content = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                content += page.extract_text() or ""  # Ensure text is added if found
            size_of_doc=len(content)
            # Append the extracted information to the data list
            data.append({
                "File Name": pdf_file,
                "Number of Pages": num_pages,
                "Metadata": metadata,
                "Count of elements": size_of_doc,
                "Content_whole":{"Number of pages": num_pages,"Content": content}
            })
    
    # Create a DataFrame with the collected data
    df = pd.DataFrame(data)
    return df
#%%
# Example usage:
# Replace 'your_directory' with the path to the folder containing the PDF files
pdf_dir = "/Users/richikghosh/Documents/Rag project/data/"
pdf_files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith('.pdf')]

# Parse all PDFs and get the DataFrame
pdf_df = parse_pdf_to_dataframe(pdf_files)

# Display the DataFrame
print(pdf_df)
################################################################################ 








# %%






# %%
###################### CHUNKING #################################
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)

pdf_df['chunk']=pdf_df.apply(
    lambda row:text_splitter.split_text(str(row['Content_whole'])),
    axis=1
)

#%%
for i in range(0,3):
    print(f"{i}'th chunk: {(pdf_df['chunk'][0][i])} and size is {len(pdf_df['chunk'][0][i])}")
#################################################################



#%%

chunkof_text=[]
for sub_chunk in pdf_df['chunk']:
    for chunk in sub_chunk:
        chunkof_text.append(chunk)



#%%





#%%
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from pinecone import Pinecone, ServerlessSpec
index_name="hybrid-search-llm-paper"

#initialize pinecone client
pc=Pinecone(api_key=api_key_pinecone)

#Create index

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='dotproduct', #sparse values are supported only on dotproduct
        spec=ServerlessSpec(cloud='aws',region='us-east-1')
    )

#%%
index=pc.Index(index_name)
index.describe_index_stats()

#%%
from dotenv import load_dotenv


from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
embedding


#%%
## Sparse Encoding
from pinecone_text.sparse import BM25Encoder
bm25encoder=BM25Encoder().default()

## tfidf values on each of the chunks
bm25encoder.fit(chunkof_text)


bm25encoder.dump("bm25_values.json")

# loading the values back from the json file
bm25encoder=BM25Encoder().load("bm25_values.json")


#%%
# Here the embeddings will be stored in PineCone

retriever=PineconeHybridSearchRetriever(embeddings=embedding,sparse_encoder=bm25encoder,index=index)

retriever.add_texts(chunkof_text)
# for batch in dataset.iter_documents(batch_size=100):
#     index.upsert(batch)

#%%
query = "What is TRP: Trained Rank Pruning for Efficient Deep Neural Networks"

model = SentenceTransformer('all-MiniLM-L6-v2')# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(vector=xq, top_k=5, include_metadata=True)
xc
#retriever.invoke("What is TRP: Trained Rank Pruning for Efficient Deep Neural Networks?")


#%%


#%%

# RAG prmpt

# template= """"
#     Answer the question based on the context below. If you can't 
#     answer the question, reply "I dont know".
#     Context: {context}
#     Question: {question}
# """
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import openai
from langchain.llms import OpenAI



# Step 1: Query Embedding (Hugging Face)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query = "What is TRP: Trained Rank Pruning for Efficient Deep Neural Networks?"
query_embedding = embedding_model.embed_query(query)


# Step 2: Retrieve Documents from Pinecone
vector_store = PineconeVectorStore(index_name='hybrid-search-llm-paper', embedding=embedding_model,pinecone_api_key=api_key_pinecone,text_key="context")
retrieved_docs = vector_store.similarity_search(query,k=3)

# Step 3: Pass Retrieved Documents to LLM (OpenAI)
# generator = pipeline('text-generation', model='gpt2', max_length=400)
# generator = pipeline("text-generation", model="gpt2",max_length=400)
# llm = HuggingFacePipeline(pipeline=generator)


# Step 3: Use OpenAI for Generation


rag_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())



# Step 4: Run the Query through the Chain
response = rag_chain.run(query)
print(response)

#%%


 # %%

######################### EMBEDDING ###################################
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Assuming you have the 'pdf_df' DataFrame with the 'Content_full_chunks' column
# which contains lists of chunked text, we will now compute embeddings for each chunk.

def generate_embeddings(text_chunks):
    return [model.encode(chunk) for chunk in text_chunks]

# Apply the embedding function to each row of 'Content_full_chunks'
pdf_df['embeddings'] = pdf_df['chunk'].apply(generate_embeddings)

# Now, 'pdf_df' has a new column 'embeddings' with a list of embeddings for each chunk in every row.

###########################################################################






#%%
len((pdf_df['chunk'][1]))

# the above chunk is embedded into a 384 dimensional vector


# %%
len(pdf_df['embeddings'][1])
#%%
embedding_data=[]
for sublist in pdf_df['embeddings'].to_list():
    for embedding in sublist:
        embedding_data.append(embedding)

chunk_data=[]
for sublist in pdf_df['chunk'].to_list():
    for chunk in sublist:
        chunk_data.append(chunk)
#%%
from langchain.vectorstores import FAISS
from langchain.schema import Document
import numpy as np
import faiss
from langchain.docstore.in_memory import InMemoryDocstore
# Convert the list of embeddings to a numpy array (required for FAISS)
embedding_data_np = np.array(embedding_data)

# Get the dimensionality of the embeddings
d = embedding_data_np.shape[1]  # assuming each embedding is a 1D vector

# Create the FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance metric (you can choose other metrics like cosine)
index.add(embedding_data_np)   # Add the embeddings to the FAISS index

# Create document objects for each chunk
documents = [Document(page_content=chunk) for chunk in chunk_data]

# Create a docstore
docstore = InMemoryDocstore(dict(enumerate(documents)))

# Create index_to_docstore_id mapping (just use the indices from FAISS as the IDs)
index_to_docstore_id = {i: str(i) for i in range(len(documents))}


# Now wrap this FAISS index using LangChain's FAISS wrapper
faiss_index = FAISS(embedding_data_np, index, docstore, index_to_docstore_id)

query_embedding = model.encode("What is TRP: Trained Rank Pruning for Efficient Deep Neural Networks?")  # Encode your query

# Search for the top 5 nearest neighbors in the FAISS index
D, I = index.search(np.array([query_embedding]), k=5)

# 'I' contains the indices of the nearest neighbors
# You can use these indices to retrieve the corresponding chunks from chunk_data
nearest_chunks = [chunk_data[i] for i in I[0]]

print(nearest_chunks)


 #%%
retriever=faiss_index.as_retriever(
    search_type="similarity", search_kwargs={"k":5}
)
#%%

#%%

import faiss
import numpy as np

# Step 1: Flatten the embeddings and keep track of document indices
all_embeddings = []
document_mapping = []
file_names = pdf_df['File Name'].tolist()

for doc_idx, embedding_list in enumerate(pdf_df['embeddings']):
    for embedding in embedding_list:
        all_embeddings.append(embedding)  # Append the embedding
        document_mapping.append(file_names[doc_idx])   # Track the file name for each embedding

# Convert the list of embeddings to a numpy array
all_embeddings = np.array(all_embeddings)
#%%
# Step 2: Create a FAISS index and add the embeddings
embedding_dim = all_embeddings.shape[1]  # Get the dimension of embeddings
index = faiss.IndexFlatL2(embedding_dim)  # Using Euclidean (L2) distance

index.is_trained
# %%
# Add all flattened embeddings to the FAISS index
index.add(all_embeddings)  
k=0
for i in range(0,100):
    k+=len(pdf_df['embeddings'][i])
print(k)
    
# %%
len(pdf_df['chunk'][0][0])


# %%

#%%

# %%







######################## Retrieval from using faiss index ###########################
from sentence_transformers import SentenceTransformer

# Step 1: Encode the question into an embedding
question = "What is TRP: Trained Rank Pruning for Efficient Deep Neural Networks?"
question_embedding = model.encode([question])

# Step 2: Search for the nearest document embeddings in the FAISS index
D, I = index.search(question_embedding, k=5)  # Find the top 5 nearest neighbors

# Step 3: Map the result indices back to file names and content
nearest_file_names = [document_mapping[idx] for idx in I[0]]
nearest_content = [pdf_df.loc[pdf_df['File Name'] == document_mapping[idx], 'Content_whole'].values[0] for idx in I[0]]

# Display the closest file names and their corresponding content
for i, (file_name, content) in enumerate(zip(nearest_file_names, nearest_content)):
    print(f"Result {i+1}:")
    print(f"File Name: {file_name}")
    print(f"Content: {content}...")  # Display the first 500 characters of the content
    print(f"Distance: {D[0][i]}")
    print()
############################
# %%
## Invoking a LLM chain

from transformers import pipeline
from langchain import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Load the smaller GPT-2 model from Hugging Face
generator = pipeline('text-generation', model='gpt2', max_length=200)

# Wrap the Hugging Face pipeline in LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "Context"],
    template="Answer the following question in a detailed manner: {question}"
)

template= """"
    Answer the question based on the context below. If you can't 
    answer the question, reply "I dont know".
    Context: {context}
    Question: {question}
"""
prompt=PromptTemplate.from_template(template)
question = "My name is Richik Ghosh"


prompt.format(context="This is a sample context",question=question)


chain= prompt | llm 
chain.input_schema.schema()
#%%
response = chain.invoke({
    "context": "The name i was given was Richik Ghosh",
    "question": "What's my name?"
})
#%%
print(response)


#%%
from langchain_core.output_parsers import StrOutputParser
chain= prompt | llm 
chain.invoke("What is TRP: Trained Rank Pruning for Efficient Deep Neural Networks?")


#%%
chain.input_schema.schema()
# %%
 
# %%
