"""Tests for the FastAPI main application."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock


# Need to mock before importing main
@pytest.fixture
def client():
    """Create a test client for the FastAPI app with mocked RAG."""
    # Mock the RAG system at module level
    with patch.dict('sys.modules', {'app.rag': MagicMock()}):
        # Create mock RAG system
        mock_rag = MagicMock()
        mock_rag.rag_system = mock_rag

        # Import and setup main module
        import sys
        sys.modules['app.rag'] = mock_rag

        from app.main import app

        # Create test client
        test_client = TestClient(app)

        yield test_client, mock_rag.rag_system

        # Cleanup
        del sys.modules['app.main']
        del sys.modules['app.rag']


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct status."""
        test_client, _ = client
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "omni-desk"

    def test_health_content_type(self, client):
        """Test health endpoint returns JSON content type."""
        test_client, _ = client
        response = test_client.get("/health")

        assert response.headers["content-type"] == "application/json"


class TestDocumentsEndpoint:
    """Test suite for /documents endpoint."""

    def test_add_documents_success(self, client):
        """Test successfully adding documents."""
        test_client, mock_rag = client
        mock_rag.add_documents = Mock()

        documents = ["Document 1 content", "Document 2 content"]
        response = test_client.post("/documents", json={"documents": documents})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["document_count"] == 2

        # Verify RAG system was called
        mock_rag.add_documents.assert_called_once_with(documents)

    def test_add_empty_documents(self, client):
        """Test adding empty documents list."""
        test_client, mock_rag = client
        mock_rag.add_documents = Mock()

        response = test_client.post("/documents", json={"documents": []})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["document_count"] == 0

        mock_rag.add_documents.assert_called_once_with([])

    def test_add_documents_missing_field(self, client):
        """Test adding documents with missing documents field."""
        test_client, _ = client
        response = test_client.post("/documents", json={})

        assert response.status_code == 422  # Validation error

    def test_add_documents_wrong_type(self, client):
        """Test adding documents with wrong type."""
        test_client, _ = client
        response = test_client.post("/documents", json={"documents": "not a list"})

        assert response.status_code == 422  # Validation error

    def test_add_single_document(self, client):
        """Test adding a single document."""
        test_client, mock_rag = client
        mock_rag.add_documents = Mock()

        documents = ["Single document content"]
        response = test_client.post("/documents", json={"documents": documents})

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 1


class TestQueryEndpoint:
    """Test suite for /query endpoint."""

    def test_query_success(self, client):
        """Test successful query."""
        test_client, mock_rag = client
        mock_rag.query.return_value = {
            "answer": "This is the answer",
            "sources": ["Source 1...", "Source 2..."]
        }

        response = test_client.post("/query", json={"question": "What is AI?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer"
        assert data["sources"] == ["Source 1...", "Source 2..."]

        mock_rag.query.assert_called_once_with("What is AI?")

    def test_query_no_documents(self, client):
        """Test query when no documents are loaded."""
        test_client, mock_rag = client
        mock_rag.query.return_value = {
            "answer": "No documents loaded. Please add documents first.",
            "sources": []
        }

        response = test_client.post("/query", json={"question": "What is AI?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "No documents loaded. Please add documents first."
        assert data["sources"] == []

    def test_query_missing_question(self, client):
        """Test query with missing question field."""
        test_client, _ = client
        response = test_client.post("/query", json={})

        assert response.status_code == 422  # Validation error

    def test_query_empty_question(self, client):
        """Test query with empty question."""
        test_client, mock_rag = client
        mock_rag.query.return_value = {
            "answer": "Some answer",
            "sources": []
        }

        response = test_client.post("/query", json={"question": ""})

        assert response.status_code == 200
        mock_rag.query.assert_called_once_with("")

    def test_query_response_model(self, client):
        """Test that query response follows QueryOutput model."""
        test_client, mock_rag = client
        mock_rag.query.return_value = {
            "answer": "Answer",
            "sources": ["Source 1"]
        }

        response = test_client.post("/query", json={"question": "Test?"})

        assert response.status_code == 200
        data = response.json()
        # Verify the response has the expected fields
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)


class TestInputModels:
    """Test suite for Pydantic input models."""

    def test_document_input_valid(self, client):
        """Test DocumentInput with valid data."""
        # Import after mocking
        import sys
        if 'app.main' in sys.modules:
            from app.main import DocumentInput
            input_data = DocumentInput(documents=["doc1", "doc2"])
            assert input_data.documents == ["doc1", "doc2"]
        else:
            pytest.skip("Cannot import DocumentInput due to module mocking")

    def test_document_input_empty_list(self, client):
        """Test DocumentInput with empty list."""
        import sys
        if 'app.main' in sys.modules:
            from app.main import DocumentInput
            input_data = DocumentInput(documents=[])
            assert input_data.documents == []
        else:
            pytest.skip("Cannot import DocumentInput due to module mocking")

    def test_query_input_valid(self, client):
        """Test QueryInput with valid data."""
        import sys
        if 'app.main' in sys.modules:
            from app.main import QueryInput
            input_data = QueryInput(question="What is AI?")
            assert input_data.question == "What is AI?"
        else:
            pytest.skip("Cannot import QueryInput due to module mocking")

    def test_query_output_valid(self, client):
        """Test QueryOutput with valid data."""
        import sys
        if 'app.main' in sys.modules:
            from app.main import QueryOutput
            output = QueryOutput(answer="Answer", sources=["Source 1"])
            assert output.answer == "Answer"
            assert output.sources == ["Source 1"]
        else:
            pytest.skip("Cannot import QueryOutput due to module mocking")

    def test_query_output_empty_sources(self, client):
        """Test QueryOutput with empty sources."""
        import sys
        if 'app.main' in sys.modules:
            from app.main import QueryOutput
            output = QueryOutput(answer="Answer", sources=[])
            assert output.sources == []
        else:
            pytest.skip("Cannot import QueryOutput due to module mocking")
