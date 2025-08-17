
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Literal
import os
import pickle

load_dotenv(override=True)

# FastMCP 서버 초기화
mcp = FastMCP(
    "RAG_Server",
    instructions="A RAG server that provides vector search, document addition, and web search capabilities."
)

# 전역 변수로 벡터스토어 관리
vector_store = None
embeddings = OpenAIEmbeddings()

def initialize_vector_store():
    """벡터 스토어를 초기화하고 PDF 문서를 로드합니다."""
    global vector_store

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "SPRI_AI_Brief_2023년12월호_F.pdf")

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

@mcp.tool()
async def vector_search(
    query: str, 
    search_type: Literal["semantic", "keyword", "hybrid"] = "semantic",
    k: int = 5
) -> str:
    """벡터 스토어에서 문서를 검색합니다."""
    global vector_store

    if vector_store is None:
        initialize_vector_store()

    if search_type == "semantic":
        results = vector_store.similarity_search(query, k=k)
    elif search_type == "keyword":
        all_docs = vector_store.similarity_search("", k=100)
        results = [doc for doc in all_docs if query.lower() in doc.page_content.lower()][:k]
    elif search_type == "hybrid":
        semantic_results = vector_store.similarity_search(query, k=k*2)
        keyword_results = [doc for doc in semantic_results if query.lower() in doc.page_content.lower()]
        results = keyword_results[:k] if keyword_results else semantic_results[:k]

    return "\n\n".join([doc.page_content for doc in results])

@mcp.tool()
async def add_document(text: str, metadata: dict = None) -> str:
    """사용자 텍스트를 벡터 스토어에 추가합니다."""
    global vector_store

    if vector_store is None:
        initialize_vector_store()

    if metadata is None:
        metadata = {"source": "user_input"}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = [Document(page_content=text, metadata=metadata)]
    splits = text_splitter.split_documents(documents)

    vector_store.add_documents(splits)

    return f"문서가 성공적으로 추가되었습니다. 총 {len(text)} 문자, {len(splits)}개 청크로 분할됨"

@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> str:
    """TavilySearch를 사용하여 웹 검색을 수행합니다."""
    tavily = TavilySearch(max_results=max_results)
    results = tavily.invoke(query)

    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"검색 결과 {i}:\n"
            f"제목: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"내용: {result.get('content', 'N/A')}\n"
        )

    return "\n".join(formatted_results)

if __name__ == "__main__":
    # 서버 초기화
    print("RAG MCP 서버를 초기화합니다...")
    initialize_vector_store()
    print("벡터 스토어 초기화 완료!")

    # MCP 서버 실행
    mcp.run(transport="stdio")
