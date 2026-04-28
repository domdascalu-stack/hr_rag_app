import json
import boto3
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=st.secrets["AWS_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    )

def hr_index():
    bedrock_client = get_bedrock_client()
    loader = PyPDFLoader('Leave-Policy-India.pdf')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(documents)
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id='amazon.titan-embed-text-v1'
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def hr_rag_response(index, question):
    bedrock_client = get_bedrock_client()
    docs = index.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Use the following HR policy excerpts to answer the question accurately.

Context:
{context}

Question: {question}

Answer:"""
    response = bedrock_client.invoke_model(
        modelId='amazon.nova-lite-v1:0',
        body=json.dumps({
           "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "max_new_tokens": 3000,
                "temperature": 0.1,
                "top_p": 0.9
            }
        }),
        contentType='application/json',
        accept='application/json'
    )
    result = json.loads(response['body'].read())
    return result['output']['message']['content'][0]['text']
