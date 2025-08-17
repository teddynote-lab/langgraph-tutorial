from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any
from rag.pdf import PDFRetrievalChain


load_dotenv(override=True)


def create_retriever() -> Any:
    """"""
    # PDF 문서를 로드합니다.
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "SPRI_AI_Brief_2023년12월호_F.pdf")
    pdf = PDFRetrievalChain([pdf_path]).create_chain()

    # retriever와 chain을 생성합니다.
    pdf_retriever = pdf.retriever

    return pdf_retriever


# Initialize FastMCP server with configuration
mcp = FastMCP(
    "Retriever",
    instructions="A Retriever that can retrieve information from the database.",
)


@mcp.tool()
async def retrieve(query: str) -> str:
    """
    Retrieves information from the document database based on the query.

    This function creates a retriever, queries it with the provided input,
    and returns the concatenated content of all retrieved documents.

    Args:
        query (str): The search query to find relevant information

    Returns:
        str: Concatenated text content from all retrieved documents
    """
    retriever = create_retriever()

    # Use the invoke() method to get relevant documents based on the query
    retrieved_docs = retriever.invoke(query)

    # Join all document contents with newlines and return as a single string
    return "\n".join([doc.page_content for doc in retrieved_docs])


if __name__ == "__main__":
    # Run the MCP server with stdio transport for integration with MCP clients
    mcp.run(transport="stdio")
