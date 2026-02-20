"""LangChain RAG pipeline for enterprise knowledge."""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


class EnterpriseRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.vectorstore = None
        self.qa_chain = None

    def add_documents(self, documents: list[str]):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = []
        for doc in documents:
            texts.extend(text_splitter.split_text(doc))
        self.vectorstore = Chroma.from_texts(texts, self.embeddings, persist_directory="./chroma_db")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def query(self, question: str) -> dict:
        if not self.qa_chain:
            return {"answer": "No documents loaded. Please add documents first.", "sources": []}
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] + "..." for doc in result.get("source_documents", [])]
        }


rag_system = EnterpriseRAG()
