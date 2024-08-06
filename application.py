import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.hub import pull
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


def load_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def embed_text():
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                       encode_kwargs={"normalize_embeddings": True})
    return embeddings


def split_text(content):
    # Specify the headers to split on in the correct format
    headers_to_split_on = [("#", "Header1"), ("##", "Header2"), ("###", "Header3")]

    # Split text: Text splitters break large Documents into smaller chunks.
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    texts = text_splitter.split_text(content)
    return texts


def main():
    load_dotenv()
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Load markdown content
    content = load_markdown(r"C:\Users\User\RAG_Implementation\pythonProject\regex-tutorial.md")

    # Split the content into smaller chunks
    texts = split_text(content)

    # Create a vectorstore
    vectorstore = FAISS.from_documents(texts, embed_text())
    
    # Save the documents and embeddings
    vectorstore.save_local("vectorstore.db")

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Load the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # Pull the retrieval QA chat prompt from the hub
    retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")

    question_answer_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "What is regex?"})
    print(response["answer"])


if __name__ == '__main__':
    main()
