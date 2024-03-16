from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from langchain_community.llms import CTransformers

def run_csv(question):
    DB_FAISS_PATH = "/Users/priya/Downloads/Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/annotations"
    loader = CSVLoader(file_path="/Users/priya/Downloads/Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/annotations/paintoff.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    # print(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local(DB_FAISS_PATH)
    # https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/file_download#huggingface_hub.hf_hub_download
    hf_hub_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GGML",
        filename="llama-2-7b-chat.ggmlv3.q8_0.bin",
        local_dir="./models"
    )
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
    # query = "What are the dimensions of the images?"
    docs = docsearch.similarity_search(question, k=3)
    print("Result", docs)
    print("Dimensions of the images:")
    if "dimensions" in question:
        for i, doc in enumerate(docs, 1):
            # Accessing content and metadata attributes directly
            content = doc.page_content
            metadata = doc.metadata
            # Extract relevant information from content and metadata
            dimensions = content.split("\n")[1].split(":")
            width, height = dimensions[0].strip(), dimensions[1].strip()
            source = metadata.get('source', 'Unknown Source')  # Assuming metadata is a dictionary
            # Print the dimensions
            print(f"Result {i}: Width = {width}, Height = {height}, Source = {source}")
            return f"Result {i}: Width = {width}, Height = {height}, Source = {source}"
    elif "total" in question or "count" in question:
        total_images = len(docs)
        print(f"type: ",type(total_images))
        print("Total number of images:", total_images)
        return f"Total number of images: {total_images}"
    else:
        return docs

# run_csv(question="What are the total number of the images?")

