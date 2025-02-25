import unittest
from unittest.mock import patch, MagicMock
import io
import sys
from agent import BmadAgent
from bmad_db import BmadDatabase

class TestBmadDatabase(unittest.TestCase):
    
    @patch('bmad_db.HuggingFaceEmbeddings')
    @patch('bmad_db.FAISS')
    def setUp(self, mock_faiss, mock_embeddings):
        # Setup mock retriever
        self.mock_retriever = MagicMock()
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.as_retriever.return_value = self.mock_retriever
        
        # Setup mocks
        mock_faiss.load_local.return_value = self.mock_vector_store
        
        # Create database instance
        self.db = BmadDatabase(db_path="test_db", verbose=False)
    
    def test_initialization(self):
        """Test that the database initializes correctly"""
        self.assertEqual(self.db.verbose, False)
        self.mock_vector_store.as_retriever.assert_called_once()
    
    def test_get_context(self):
        """Test context retrieval"""
        # Mock the retriever to return test documents
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Test content 1"
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Test content 2"
        self.mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        
        # Get context
        context = self.db.get_context("test query")
        
        # Verify retriever was called
        self.mock_retriever.invoke.assert_called_once_with("test query")
        
        # Verify context formatting
        self.assertIn("Document 1:", context)
        self.assertIn("Test content 1", context)
        self.assertIn("Document 2:", context)
        self.assertIn("Test content 2", context)

class TestBmadAgent(unittest.TestCase):
    
    @patch('agent.OpenAI')
    @patch('agent.BmadDatabase')
    def setUp(self, mock_db_class, mock_openai):
        # Setup mock database
        self.mock_db = MagicMock()
        mock_db_class.return_value = self.mock_db
        
        # Setup OpenAI client mock
        self.mock_openai_client = MagicMock()
        mock_openai.return_value = self.mock_openai_client
        
        # Create agent instance
        self.agent = BmadAgent(model_name="test-model", verbose=False)
    
    def test_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertEqual(self.agent.model_name, "test-model")
        self.assertEqual(self.agent.verbose, False)
    
    def test_generate_response(self):
        """Test response generation"""
        # Mock OpenAI response
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Generate response
        response = self.agent.generate_response("test query", "test context")
        
        # Verify OpenAI client was called
        self.mock_openai_client.chat.completions.create.assert_called_once()
        
        # Verify response
        self.assertEqual(response, "Test response")
    
    def test_ask(self):
        """Test the ask method"""
        # Setup mocks
        self.mock_db.get_context.return_value = "mocked context"
        
        # Mock generate_response
        self.agent.generate_response = MagicMock(return_value="mocked response")
        
        # Ask a question
        response = self.agent.ask("test question")
        
        # Verify methods were called
        self.mock_db.get_context.assert_called_once_with("test question")
        self.agent.generate_response.assert_called_once_with("test question", "mocked context")
        
        # Verify response
        self.assertEqual(response, "mocked response")
    
    @patch('builtins.input', side_effect=["test query", "exit"])
    @patch('builtins.print')
    def test_run_interactive(self, mock_print, mock_input):
        """Test interactive mode"""
        # Setup mock ask method
        self.agent.ask = MagicMock(return_value="test response")
        
        # Run interactive mode
        self.agent.run_interactive()
        
        # Verify ask was called
        self.agent.ask.assert_called_once_with("test query")
        
        # Verify exit message
        mock_print.assert_any_call("Goodbye!")

class TestIntegration(unittest.TestCase):
    """Integration tests that require actual dependencies"""
    
    @unittest.skip("Skip integration test that requires dependencies")
    def test_integration_query(self):
        """Test a simple query with actual dependencies"""
        agent = BmadAgent(model_name="gpt-3.5-turbo")
        response = agent.ask("What is a quadrupole in Bmad?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()