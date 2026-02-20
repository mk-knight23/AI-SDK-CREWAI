"""Tests for the EnterpriseRAG system."""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestEnterpriseRAG:
    """Test suite for EnterpriseRAG class."""

    @pytest.fixture
    def rag(self):
        """Create a fresh EnterpriseRAG instance for each test."""
        # Import here after conftest.py has set up mocks
        from app.rag import EnterpriseRAG
        return EnterpriseRAG()

    def test_init(self, rag):
        """Test that EnterpriseRAG initializes correctly."""
        assert rag.embeddings is not None
        assert rag.llm is not None
        assert rag.vectorstore is None
        assert rag.qa_chain is None

    @patch('app.rag.Chroma')
    @patch('app.rag.RetrievalQA')
    @patch('app.rag.CharacterTextSplitter')
    def test_add_documents(self, mock_splitter_class, mock_retrieval_qa, mock_chroma, rag):
        """Test adding documents to the RAG system."""
        # Setup mocks
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter

        mock_vectorstore = Mock()
        mock_chroma.from_texts.return_value = mock_vectorstore

        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

        # Test adding documents
        documents = ["This is document 1", "This is document 2"]
        rag.add_documents(documents)

        # Verify ChromaDB was called
        mock_chroma.from_texts.assert_called_once()
        call_args = mock_chroma.from_texts.call_args
        # 2 documents x 2 chunks each = 4 chunks total
        assert call_args[0][0] == ["chunk1", "chunk2", "chunk1", "chunk2"]
        assert call_args[1]['persist_directory'] == "./chroma_db"

        # Verify QA chain was created
        mock_retrieval_qa.from_chain_type.assert_called_once()
        assert rag.vectorstore == mock_vectorstore
        assert rag.qa_chain == mock_qa_chain

    @patch('app.rag.Chroma')
    @patch('app.rag.RetrievalQA')
    @patch('app.rag.CharacterTextSplitter')
    def test_add_documents_empty_list(self, mock_splitter_class, mock_retrieval_qa, mock_chroma, rag):
        """Test adding empty document list."""
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = []
        mock_splitter_class.return_value = mock_splitter

        mock_vectorstore = Mock()
        mock_chroma.from_texts.return_value = mock_vectorstore

        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

        # Test adding empty documents
        rag.add_documents([])

        # Should still call Chroma with empty list
        mock_chroma.from_texts.assert_called_once()
        call_args = mock_chroma.from_texts.call_args
        assert call_args[0][0] == []

    @patch('app.rag.Chroma')
    @patch('app.rag.RetrievalQA')
    @patch('app.rag.CharacterTextSplitter')
    def test_add_documents_with_chunking(self, mock_splitter_class, mock_retrieval_qa, mock_chroma, rag):
        """Test that documents are properly chunked before storage."""
        mock_splitter = Mock()
        mock_splitter.split_text.side_effect = lambda x: [x[:5], x[5:]] if len(x) > 5 else [x]
        mock_splitter_class.return_value = mock_splitter

        mock_vectorstore = Mock()
        mock_chroma.from_texts.return_value = mock_vectorstore

        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

        # Add a long document
        long_doc = "This is a very long document that should be chunked"
        rag.add_documents([long_doc])

        # Verify splitter was called with correct parameters
        mock_splitter_class.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
        mock_splitter.split_text.assert_called_with(long_doc)

    def test_query_without_documents(self, rag):
        """Test querying before documents are added."""
        result = rag.query("What is AI?")

        assert result["answer"] == "No documents loaded. Please add documents first."
        assert result["sources"] == []

    def test_query_with_documents(self, rag):
        """Test querying with documents loaded."""
        # Setup mock QA chain
        mock_qa_chain = Mock()
        mock_source_doc = Mock()
        mock_source_doc.page_content = "This is a source document with some content"

        mock_qa_chain.invoke.return_value = {
            "result": "AI is artificial intelligence",
            "source_documents": [mock_source_doc]
        }
        rag.qa_chain = mock_qa_chain

        result = rag.query("What is AI?")

        assert result["answer"] == "AI is artificial intelligence"
        assert len(result["sources"]) == 1
        assert result["sources"][0].endswith("...")
        assert result["sources"][0].startswith("This is a source document")

        # Verify the QA chain was invoked
        mock_qa_chain.invoke.assert_called_once_with({"query": "What is AI?"})

    def test_query_with_multiple_sources(self, rag):
        """Test querying returns multiple sources."""
        mock_qa_chain = Mock()

        mock_doc1 = Mock()
        mock_doc1.page_content = "Source document 1 content here"
        mock_doc2 = Mock()
        mock_doc2.page_content = "Source document 2 content here"
        mock_doc3 = Mock()
        mock_doc3.page_content = "Source document 3 content here"

        mock_qa_chain.invoke.return_value = {
            "result": "Comprehensive answer",
            "source_documents": [mock_doc1, mock_doc2, mock_doc3]
        }
        rag.qa_chain = mock_qa_chain

        result = rag.query("Complex question?")

        assert result["answer"] == "Comprehensive answer"
        assert len(result["sources"]) == 3

    def test_query_truncates_long_sources(self, rag):
        """Test that long source documents are truncated."""
        mock_qa_chain = Mock()

        mock_doc = Mock()
        mock_doc.page_content = "x" * 500  # Very long content

        mock_qa_chain.invoke.return_value = {
            "result": "Answer",
            "source_documents": [mock_doc]
        }
        rag.qa_chain = mock_qa_chain

        result = rag.query("Question?")

        # Source should be truncated to 200 chars + "..."
        assert len(result["sources"][0]) == 203  # 200 chars + "..."
        assert result["sources"][0].endswith("...")

    def test_query_no_source_documents(self, rag):
        """Test querying when no source documents are returned."""
        mock_qa_chain = Mock()
        mock_qa_chain.invoke.return_value = {
            "result": "Answer without sources",
            "source_documents": []
        }
        rag.qa_chain = mock_qa_chain

        result = rag.query("Question?")

        assert result["answer"] == "Answer without sources"
        assert result["sources"] == []

    def test_query_missing_source_documents_key(self, rag):
        """Test querying when source_documents key is missing."""
        mock_qa_chain = Mock()
        mock_qa_chain.invoke.return_value = {
            "result": "Answer without sources key"
        }
        rag.qa_chain = mock_qa_chain

        result = rag.query("Question?")

        assert result["answer"] == "Answer without sources key"
        assert result["sources"] == []
