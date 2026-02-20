"""Pytest configuration and fixtures."""
import sys
from unittest.mock import MagicMock, Mock

# Mock all langchain modules before any imports
langchain_mock = MagicMock()
langchain_openai_mock = MagicMock()
langchain_community_mock = MagicMock()
langchain_text_splitter_mock = MagicMock()
langchain_chains_mock = MagicMock()

# Setup vectorstores submodule properly
langchain_community_vectorstores_mock = MagicMock()
langchain_community_mock.vectorstores = langchain_community_vectorstores_mock

sys.modules['langchain'] = langchain_mock
sys.modules['langchain_openai'] = langchain_openai_mock
sys.modules['langchain_community'] = langchain_community_mock
sys.modules['langchain_community.vectorstores'] = langchain_community_vectorstores_mock
sys.modules['langchain.text_splitter'] = langchain_text_splitter_mock
sys.modules['langchain.chains'] = langchain_chains_mock

# Setup mock classes with proper attributes
langchain_openai_mock.OpenAIEmbeddings = Mock
langchain_openai_mock.ChatOpenAI = Mock

# Chroma needs to be a class with from_texts method
mock_chroma_class = Mock()
langchain_community_vectorstores_mock.Chroma = mock_chroma_class

langchain_text_splitter_mock.CharacterTextSplitter = Mock

# RetrievalQA needs to be a class with from_chain_type method
mock_retrieval_qa_class = Mock()
langchain_chains_mock.RetrievalQA = mock_retrieval_qa_class
