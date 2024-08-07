import os
import time

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.hub import pull
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from openai import RateLimitError


def load_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def embed_text():
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

    content = load_markdown(r"C:\Users\User\RAG_Implementation\pythonProject\regex-tutorial.md")

    texts = split_text(content)

    vectorstore = FAISS.from_documents(texts, embed_text())

    vectorstore.save_local("vectorstore.db")

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")

    question_answer_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    query = "What is regex?"
    response = None
    retry_attempts = 5
    # This limits the amount of attempts with a backoff-logic
    for attempt in range(retry_attempts):
        try:

            response = rag_chain.invoke({"input": query})  # this is line is the source of error.
            # Not sure if the error would disappear, if had more tokens.
            break
        except RateLimitError:
            wait_time = (2 ** attempt)  # in seconds 1,2,4,8 ....
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            if wait_time > 4:
                break

    if response:
        print(response["answer"])
    else:
        print("Failed to get a response after several attempts.")


if __name__ == '__main__':
    main()
