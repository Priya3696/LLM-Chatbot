from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import PodSpec
from pinecone import Pinecone
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from pinecone import PodSpec
import os
import tiktoken
import pinecone
load_dotenv(find_dotenv(), override=True)

os.environ["PINECONE_API_KEY"] = '9c9b0591-23b8-4265-91a6-dab98f3ba418'
os.environ["OPENAI_API_KEY"] = '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'

def load_document(file):
  print(f'Loading{file}')
  loader = PyPDFLoader(file)
  data = loader.load()
  return data

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    # enc = tiktoken.encoding_for_model('text-embedding-3-small')
    enc = tiktoken.get_encoding('cl100k_base')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
'''''
def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client

    os.environ["PINECONE_API_KEY"] = "9c9b0591-23b8-4265-91a6-dab98f3ba418"
    OPENAI_API_KEY = "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"

    pc = pinecone.Pinecone(api_key='9c9b0591-23b8-4265-91a6-dab98f3ba418')
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536, api_key=OPENAI_API_KEY)

    # loading from existing index
    if index_name in pc.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index
        print(f'Creating index {index_name} and embeddings ...', end='')
        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(
                environment='gcp-starter'
            )
        )
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store


def delete_pinecone_index(index_name='all', api_key="9c9b0591-23b8-4265-91a6-dab98f3ba418"):
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536, api_key='config.ini')

    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


def ask_and_get_answer(vector_store, q, k=3):
    OPENAI_API_KEY = "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"

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer
    

'''
def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):

    OPENAI_API_KEY = "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"

    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=OPENAI_API_KEY)

    # Create a Chroma vector store using the provided text chunks and embedding model,
    # configuring it to save data to the specified directory
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

    return vector_store  # Return the created vector store

def load_embeddings_chroma(persist_directory='./chroma_db'):
    OPENAI_API_KEY = "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"

    # Instantiate the same embedding model used during creation
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536, api_key=OPENAI_API_KEY)

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vector_store  # Return the loaded vector store

# # Loading the pdf document into LangChain
# data = load_document('/Users/priya/Bot_Builder/ChatBot/Dataset_HandsonGenAI-task/LiteratureData/2301.03228.pdf')
# # Splitting the document into chunks
# chunks = chunk_data(data, chunk_size=150)
# # Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
# vector_store = create_embeddings_chroma(chunks)

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    OPENAI_API_KEY = "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"

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer

# # Asking questions
# q = 'What is the whole document about?'
# answer = ask_and_get_answer(vector_store, q)
# print(answer)

def run(question):
    # data = load_document('./Dataset_HandsonGenAI-task/LiteratureData/2301.03228.pdf')
    # chunks = chunk_data(data)[:100]  # Selecting only the first 150 chunks
    # print(len(chunks))
    # print(chunks[10].page_content)
    # print_embedding_cost(chunks)
    # index_name = 'askadocument'
    # vector_store = insert_or_fetch_embeddings(index_name, chunks)
    # answer = ask_and_get_answer(vector_store, question)
    # print(answer)
    # return answer
    # index_name = 'priya'
    # vector_store = insert_or_fetch_embeddings(index_name, chunks)
    # pc = Pinecone(api_key="9c9b0591-23b8-4265-91a6-dab98f3ba418")
    # # Print the list of existing indexes
    # existing_indexes = pc.list_indexes()
    # print("Existing indexes:", existing_indexes)
    # q = 'What is the whole document about?'
    # answer = ask_and_get_answer(vector_store, q)
    # print(answer)

    # Loading the pdf document into LangChain
    data = load_document('./Dataset_HandsonGenAI-task/LiteratureData/2301.03228.pdf')
    # Splitting the document into chunks
    chunks = chunk_data(data, chunk_size=150)
    # Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
    vector_store = create_embeddings_chroma(chunks)
    # Asking questions
    answer = ask_and_get_answer(vector_store, question)
    print(answer)
    return answer['result']



# run(question='What is the whole document about?')