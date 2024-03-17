import uuid
from dotenv import load_dotenv, find_dotenv
import os
import base64
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.csv import partition_csv
import pytesseract
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.chains.router.multi_retrieval_qa import MultiRetrievalQAChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from unstructured.partition.html import partition_html
from langchain_community.document_loaders import UnstructuredHTMLLoader


load_dotenv(find_dotenv(), override=True)
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
open_api_key = os.getenv("OPENAI_API_KEY")
surface_defect_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/annotations'
blade_turbine_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Wind turbines and blades/annotations'
python_file_path = './Dataset_HandsonGenAI-task/PythonScripts'
wesgraph_path = './Dataset_HandsonGenAI-task'
other_img_path = './Dataset_HandsonGenAI-task/Image_SupervisedInput/Surface defects/images'
now_cwd = os.getcwd()
input_path = os.path.join(now_cwd, 'Dataset_HandsonGenAI-task/LiteratureData')
output_path = os.path.join(now_cwd, 'output')
text_elements = []
table_elements = []
image_elements = []
python_script_elements = []
chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024, api_key=open_api_key)
chain_gpt_4_vision = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024, api_key=open_api_key)
table_summaries = []
text_summaries = []
image_summaries = []
other_image_summaries = []
python_summaries = []
csv_summaries = []
json_summaries = []

def partition_csv_elements(input_path, tag):
    # {'error': {
    #     'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 45361 tokens. Please reduce the length of the messages.",
    #     'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
    raw_csv_elements = None
    output_path_name = os.path.join(now_cwd, 'output')
    output_file_name = os.path.join(output_path_name, 'surfacecsv.html')
    raw_csv_elements_list =[]
    # csv_element_html = []
    files = os.listdir(input_path)
    print(files)
    for file in files:
        name, extension = os.path.splitext(file)
        if extension == ".csv":
            file_name = os.path.join(input_path, file)
            raw_csv_elements = partition_csv(filename=file_name)
            # raw_csv_elements_list.extend(raw_csv_elements)
    print(raw_csv_elements[0].metadata.text_as_html)
    output_path_name = os.path.join(now_cwd, 'output')
    output_file_name = output_path_name + "/" + tag + "csv.html"
    with open(output_file_name, 'w') as f:
        # Iterate over the list
        for item in raw_csv_elements:
            # Write each item on a new line
            f.write(f'{item}\n')
    return raw_csv_elements[0].metadata.text_as_html

def partition_html_file(file_path):
    elements = partition_html(filename=file_path)
    print(elements)
    return elements

def partion_pdf_elements(input_path):
    raw_pdf_elements = None
    now_cwd = os.getcwd()
    # input_path = os.path.join(now_cwd, 'Dataset_HandsonGenAI-task/LiteratureData')
    files = os.listdir(input_path)
    for file in files:
        name, extension = os.path.splitext(file)
        file_name = os.path.join(input_path, file)
        print(f"file name of pdf: {file_name}")
        raw_pdf_elements = partition_pdf(filename=file_name,
                                             extract_images_in_pdf=True,
                                             infer_table_structure=True,
                                             strategy="fast",
                                             max_characters=4000,
                                             new_after_n_chars=3800,
                                             combine_text_under_n_chars=2000,
                                             image_output_dir_path=os.path.join(now_cwd, 'output'))
        print(raw_pdf_elements)
    return raw_pdf_elements

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_all_elements(pdf_elements, output_path, table_elements=table_elements,
                     text_elements= text_elements, image_elements=image_elements):
    for element in pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text_elements.append(element)
        elif 'Table' in str(type(element)):
            table_elements.append(element)
    table_elements = [i.text for i in table_elements]
    text_elements = [i.text for i in text_elements]
    print(len(table_elements))
    print(len(text_elements))
    for image_file in output_path:
        if image_file.endswith(('.png','.jpg', '.jpeg')):
            image_path = os.path.join(output_path, image_file)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
    print(len(image_elements))
    return table_elements, text_elements, image_elements


def load_encode_images(file_path):
    other_images = []
    for image_file in file_path:
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, image_file)
            encoded_image = encode_image(image_path)
            other_images.append(encoded_image)
    print(len(other_images))
    return other_images


def summarize_text(text_element):
    prompt= f"summarize the following text: \n\n {text_element} \n\n Summary: "
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content


def summarize_table(table_element):
    prompt = f"summarize the following text: \n\n {table_element} \n\n Summary: "
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content


def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="you are a bot that is good at analyzing images"),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contentsof this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64, {encoded_image}"
                },
            },
        ])
    ]
    response = chain_gpt_4_vision.invoke(prompt)
    return response.content

def summarize_all_tables(table_elements, table_summaries=table_summaries):
    for i, te in enumerate(table_elements[0:2]):
        summary = summarize_table(te)
        table_summaries.append(summary)
        print(f"{i}th element of tables processed")

def summarize_all_text(text_elements, text_summaries=text_summaries):
    for i, te in enumerate(text_elements[0:2]):
        summary = summarize_text(te)
        text_summaries.append(summary)
        print(f"{i}th element of text processed")

def summarize_all_python_scripts(python_elements, python_summaries=python_summaries):
    for i, te in enumerate(python_elements):
        summary = summarize_text(te)
        python_summaries.append(summary)
        print(f"{i}th element of python code processed")


def summarize_all_csv(csv_elements, csv_summaries=csv_summaries):
    for i, te in enumerate(csv_elements):
        summary = summarize_text(te)
        csv_summaries.append(summary)
        print(f"{i}th element of csv processed")


def summarize_all_json(json_elements, json_summaries=json_summaries):
    for i, te in enumerate(json_elements):
        summary = summarize_text(te)
        json_summaries.append(summary)
        print(f"{i}th element of json processed")


def summarize_all_image(image_elements, image_summaries=image_summaries):
    for i, te in enumerate(image_elements[0:2]):
        summary = summarize_image(te)
        image_summaries.append(summary)
        print(f"{i}th element of image processed")

def summarize_other_images(other_images, other_image_summaries=other_image_summaries):
    for i, te in enumerate(other_images):
        summary = summarize_image(te)
        other_image_summaries.append(summary)
        print(f"{i}th element of other image processed")


def init_retriever():
    # vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
    vectorstore = Chroma(collection_name="full_documents", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    return retriever, id_key

def add_documents_to_retriever(retriever, id_key, summaries, original_contents):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, original_contents)))
    return retriever


def add_sub_docs_to_retriever(retriever, doc_ids, sub_docs):
    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, sub_docs)))
    return retriever



def run(user_prompt):
    csv_elements_list =[]
    retriever, key = init_retriever()
    if '@pdf' in user_prompt:
        pdf_elements = partion_pdf_elements(input_path)
        tables, texts, images = get_all_elements(pdf_elements=pdf_elements, output_path=output_path)
        summarize_all_text(texts)
        summarize_all_tables(tables)
        summarize_all_image(images)
        if not texts:
            print("texts are empty")
        else:
            retriever = add_documents_to_retriever(retriever, key, text_summaries, texts)
        if not tables:
            print("tables are empty")
        else:
            retriever = add_documents_to_retriever(retriever, key, table_summaries, tables)
        if not images:
            print("images are empty")
        else:
            retriever = add_documents_to_retriever(retriever, key, image_summaries, images)
    elif '@csv' in user_prompt:
        ### intial one time run to generate html files
        # surface_csv_elements = partition_csv_elements(surface_defect_path, "surface")
        # csv_elements_list.append(surface_csv_elements)
        # This model's maximum context length is 16385 tokens. However, your messages resulted in 45361 tokens
        # blade_csv_elements = partition_csv_elements(blade_turbine_path, "blade")
        # csv_elements_list.append(blade_csv_elements)
        # print(csv_elements_list)
        #### model unable to provide answer ####
        # surface_csv_html = partition_html_file("./output/surfacecsv.html")
        # csv_elements_list.append(surface_csv_html)
        # blade_csv_html = partition_html_file("./output/bladecsv.html")
        # csv_elements_list.append(blade_csv_html)
        surface_loader = UnstructuredHTMLLoader("./output/surfacecsv.html")
        csv_elements_list.extend(surface_loader.load())
        # This model's maximum context length is 16385 tokens. However, your messages resulted in 17118 tokens
        #commenting below due to token exceed issue
        # blade_loader = UnstructuredHTMLLoader("./output/bladecsv.html")
        # csv_elements_list.extend(blade_loader.load())
        summarize_all_csv(csv_elements_list)
        add_documents_to_retriever(retriever, key, csv_summaries, csv_elements_list)
    retriever.get_relevant_documents(user_prompt)
    template = """Answer the question based only on the following context, which can include text, images and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    answer = chain.invoke(user_prompt)
    print(answer)
    return answer


# run(' @pdf explain about Shallow Neural Network') #pdf question
# run('@csv What is the jpg related to crack?') # csv question
