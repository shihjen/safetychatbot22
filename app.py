# import dependencies 
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# get the api key from .env file
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit UI
st.title('Safety Chatbot22')
st.write('Hello, I am Bot 22. I can assist you if you have any safety-related questions such as chemical, biological or radiation safety when working in the NUS laboratories. I am currently unable to access safety manuals on my own, but click the [Activate Me] button below and I will be ready to answer your questions.')

# initialize the llm model --- llama3 
llm=ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# create a prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
"""
)

# define a function which load the data, split data into chunks, perform text embeddings and store the embeddings in a vector database
def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=huggingFace_embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':False})
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.loader=PyPDFDirectoryLoader('./data')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI for user to input their question
prompt1 = st.chat_input('Please enter your question')

# Streamlit UI for user to initialize the data loading, text splitting, embedding and create vector database
if st.button('Activate Me'):
    st.write('Give me some time to read the safety manual. I will tell you when I have finished reading....')
    vector_embedding()
    st.write('I have finished reading the safety manuals. You may ask me your question now.')

# data retrieval and answer user's query based on provided context
if prompt1:
    st.write('User:', prompt1)
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    with st.expander('These are the references I have read for answering your question:'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------------------------')
