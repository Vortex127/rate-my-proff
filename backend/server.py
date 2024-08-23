# from langchain import hub
# from langchain import PromptTemplate
# from langchain.docstore.document import Document
# from langchain.document_loaders import WebBaseLoader
# from langchain.schema import StrOutputParser
# from langchain.schema.prompt_template import format_document
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_pinecone import Pinecone
# from langchain_community.document_loaders import JSONLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from flask import Flask
# import os


# from pinecone import Pinecone as pc
# from pinecone import PodSpec

# def url():
#     #when URL
#     loader = WebBaseLoader("https://www.andrewng.org/about/")
#     docs = loader.load()
#     return docs

# def json():
#     #when json
#     loader = JSONLoader(
#         file_path='./test.json',
#         jq_schema='.professors[]',
#         text_content=False)

#     docs = loader.load()
#     return docs

# def format_docs(docs):
#   return "\n\n".join(doc.page_content for doc in docs)

# def pine_index(docs,gemini_embeddings):
#     pine_client = pc()
#     index_name = "langchain-demo"
#     if index_name not in pine_client.list_indexes().names():
#         print("Creating index")
#         pine_client.create_index(name=index_name,
#                         metric="cosine",
#                         dimension=768,
#                         spec=PodSpec(
#                             environment="gcp-starter",
#                             pod_type="starter",
#                             pods=1)
#         )
#         print(pine_client.describe_index(index_name))

#     vectorstore = Pinecone.from_documents(docs,gemini_embeddings, index_name=index_name)
#     #test vector store
#     retriever = vectorstore.as_retriever()
#     print(len(retriever.invoke("MMLU")))
    
#     return retriever

# def gemini(retriever,question):
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temprature=0.7,top_p=0.85)
#     llm_prompt_template = """You are an assistant for question-answering tasks.
#     Use the following context to answer the question.
#     If you don't know the answer, just say that you don't know.
#     Use five sentences maximum and keep the answer concise.

#     Question: {question}
#     Context: {context}
#     Answer:"""

#     llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    
#     rag_chain = (
#         {"context":retriever | format_docs,"question": RunnablePassthrough()}
#         | llm_prompt
#         | llm
#         | StrOutputParser()
#     )
#     print(rag_chain.invoke(question))

# if __name__ == "__main__":
#     text_content =  json()
#     # text_content_1 = text_content.split("code, audio, image and video.",1)[1]
#     # final_text = text_content_1.split("Cloud TPU v5p",1)[0]
#     final_text = text_content
#     docs = [Document(page_content = str(final_text),metadata={"source":"local"})]
    
#     os.environ['PINECONE_API_KEY'] = 
#     os.environ['GOOGLE_API_KEY'] = 

#     gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     question = input("What is your query:  ")
#     retriever = pine_index(docs,gemini_embeddings)
#     gemini(retriever,question)

from flask import Flask, request, jsonify
import os
from langchain import hub
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import JSONLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone as pc
from pinecone import PodSpec

app = Flask(__name__)

def load_json():
    loader = JSONLoader(
        file_path='./test.json',
        jq_schema='.professors[]',
        text_content=False)
    docs = loader.load()
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def pine_index(docs, gemini_embeddings):
    pine_client = pc()
    index_name = "langchain-demo"
    if index_name not in pine_client.list_indexes().names():
        pine_client.create_index(name=index_name,
                                 metric="cosine",
                                 dimension=768,
                                 spec=PodSpec(
                                     environment="gcp-starter",
                                     pod_type="starter",
                                     pods=1))
        print(pine_client.describe_index(index_name))

    vectorstore = Pinecone.from_documents(docs, gemini_embeddings, index_name=index_name)
    retriever = vectorstore.as_retriever()
    return retriever

def gemini_answer(retriever, question):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, top_p=0.85)
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    docs = load_json()
    formatted_docs = [Document(page_content=str(docs), metadata={"source": "local"})]

    os.environ['PINECONE_API_KEY'] = 'your key'
    os.environ['GOOGLE_API_KEY'] = 'your key'

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    retriever = pine_index(formatted_docs, gemini_embeddings)
    answer = gemini_answer(retriever, question)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
