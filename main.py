from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.chains import ConversationalRetrievalChain
import os
import json
from pathlib import Path
from pprint import pprint

load_dotenv(find_dotenv(), override=True)

pinecone_key = os.getenv("PINECONE_API_KEY")
open_api_key = os.getenv("OPENAI_API_KEY")
pdf_dir_path = './Dataset_HandsonGenAI-task/LiteratureData/'
surface_defect_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/annotations'
blade_turbine_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Wind turbines and blades/annotations'
python_file_path = './Dataset_HandsonGenAI-task/PythonScripts'
crack_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/images/crack'
erosion_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/images/erosion'
good_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/images/good'
paintoff_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/images/paintoff'
scratch_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/images/scratch'
blade_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Wind turbines and blades/images/blade'
turbine_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Wind turbines and blades/images/turbine'
wesgraph_path = './Dataset_HandsonGenAI-task'

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(directory, expression=None):
    documents = []
    files = os.listdir(directory)
    for file in files:
        name, extension = os.path.splitext(file)
        if extension == '.pdf':
            pdf_path = os.path.join(directory, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif extension == '.csv' and name == 'crack':
            csv_path = os.path.join(directory, file)
            loader = CSVLoader(csv_path)
            documents.extend(loader.load())
        elif extension == '.py':
            script_path = os.path.join(directory, file)
            loader = TextLoader(script_path)
            documents.extend(loader.load())
        elif extension == '.jpg':
            img_path = os.path.join(directory, file)
            loader = UnstructuredImageLoader(img_path)
            documents.extend(loader.load())
        elif extension == '.json':
            json_file_path = os.path.join(wesgraph_path, file)
            loader = JSONLoader(
                file_path=json_file_path,
                jq_schema=expression,
                text_content=False)
            documents.extend(loader.load())
    return documents


def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def json_chunk_data(data, chunk_size=300):
    splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
    json_chunks = splitter.split_json(json_data=data)
    print(json_chunks)
    return json_chunks

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536, api_key=OPENAI_API_KEY)
    print(chunks)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=open_api_key)
    # Create a Chroma vector store using the provided text chunks and embedding model,
    # configuring it to save data to the specified directory
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store  # Return the created vector store


def create_image_embeddings(img_raw_data):
    model_name = "ViT-B-32"
    checkpoint = "laion2b_s34b_b79k"
    clip_embd = OpenCLIPEmbeddings(model_name=model_name, checkpoint=checkpoint)
    print(img_raw_data)




def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=open_api_key)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    matching_docs = vector_store.similarity_search(q, k)
    answer = chain.run(input_documents=matching_docs, question=q)
    return answer

def qa(vector_store):
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vector_store.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )
    answer = chain.invoke()

def run_server(question):
    final_raw_data = []
    img_raw_data = []
    json_raw_data = []
    chunks_list = []
    if '@json' in question:
        wesgraph_json_data = load_document(wesgraph_path, expression='.nodes | map(.properties)')
        final_raw_data.extend(wesgraph_json_data)
    else:
        python_scripts_data = load_document(python_file_path)
        final_raw_data.extend(python_scripts_data)
    # Commented image embed due to openclip embeddings experimental error (not sure) need to look into it
    # crack_img_data = load_document(crack_img_path)
    # img_raw_data.extend(crack_img_data)
    # erosion_img_data = load_document(erosion_img_path)
    # img_raw_data.extend(erosion_img_data)
    # good_img_data = load_document(good_img_path)
    # img_raw_data.extend(good_img_data)
    # paintoff_img_data = load_document(paintoff_img_path)
    # img_raw_data.extend(paintoff_img_data)
    # blade_img_data = load_document(blade_img_path)
    # img_raw_data.extend(blade_img_data)
    # turbine_img_data = load_document(turbine_img_path)
    # img_raw_data.extend(turbine_img_data)
    # # Splitting the document into chunks
    # if 'image' in question or 'picture' in question or 'photo' in question or 'jpg' in question:
    #     img_store = create_image_embeddings(json_raw_data)
    chunks = chunk_data(final_raw_data, chunk_size=10000)
    # # Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
    vector_store = create_embeddings_chroma(chunks)
    # # Asking questions
    cleaned_question = question.replace("@json ", "")
    cleaned_question = cleaned_question.replace("@pdf ", "")
    answer = ask_and_get_answer(vector_store, cleaned_question)
    print(answer)
    return answer


# run_server('what is the formula for calculating the yawfatigue?') #python question
# run_server('What is the total number of the images related to crack?') # csv question
# run_server('@json what are the properties of monopile?') #json question
# run_server('@json what are the authors in edges?') #json question