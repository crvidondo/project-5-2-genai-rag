import streamlit as st
import os
import pandas as pd
import numpy as np
import evaluate
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import display, Markdown

# Title and description for the Streamlit app
st.title("Allergy Information Assistant")
st.write("This app is trained with documentation from the American College of Allergy, Asthma and Immunology to answer questions about allergies, symptoms and management strategies.")


st.header("Have you got any question?")
user_question = st.text_input("")

if st.button("Get my answer!"):
    st.write("Analyzing question...")

    # Load the document
    document_dir = "./data"
    filename = "allergies-doc.pdf"
    file_path = os.path.join(document_dir, filename)



    # Load the document as pages
    pages = PyPDFLoader(file_path).load_and_split()

    API_KEY = ""



    # Create the embeddings function
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key = API_KEY)

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

    # Split the document into chunks
    chunks = text_splitter.split_documents(pages)

    # Load it into Chroma
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")


    ### CONNECTING TO LLM

    # Retrieve relevant documents based on the user's question.
    docs = db.similarity_search(user_question, k=3)

    # Build a function to create a context paragraph for the Prompt
    def _get_document_context(docs):
        context = '\n'
        for doc in docs:
            context += '\nContext:\n'
            context += doc.page_content + '\n\n'
        return context

    # Construct the prompt for the LLM including the context based on the results from the query
    def generate_prompt(user_question, docs):
        prompt = f"""
        INTRODUCTION
        You are a knowledgeable assistant trained to answer questions about allergies, symptoms, and management strategies. Your responses should be clear, concise, and focused on accurate information.

        The user asked: "{user_question}"

        CONTEXT
        Technical documentation for allergies, symptoms, and management of allergen ingestion:
        '''
        {_get_document_context(docs)}
        '''

        RESTRICTIONS
        Always refer to products or allergens by their specific names as mentioned in the documentation.
        Stick to facts and provide clear, evidence-based responses; avoid opinions or interpretations.
        Only respond if the answer can be found within the context. If not, let the user know that the information is not available.
        Do not engage in topics outside allergies, symptoms, and related health matters. Avoid humor, sensitive topics, and speculative discussions.
        If the user’s question lacks sufficient details, request clarification rather than guessing the answer.

        TASK
        Provide a direct answer based on the user’s question, if possible.
        Guide the user to relevant sections of the documentation if additional context is needed.
        Format the response in Markdown format.

        RESPONSE STRUCTURE:
        '''
        # [Answer Title]
        [answer text]
        '''

        CONVERSATION:
        User: {user_question}
        Agent:
        """
        return prompt

    prompt = generate_prompt(user_question, docs)


    # Initialize an OpenAI Assistant that responds to user's question
    client = OpenAI(api_key = API_KEY)

    messages = [{'role':'user', 'content':prompt}]
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 200}
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)


    answer = completion.choices[0].message.content

    print(f'User asked: {user_question}:\n')
    print(answer)

    st.write(answer)
