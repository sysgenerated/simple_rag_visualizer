import os
import google.generativeai as genai
import numpy as np
import streamlit as st

class EmbeddingManager:
    def __init__(self):
        # Ensure the Google AI API key is set
        if 'GOOGLE_API_KEY' not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = 'models/text-embedding-004'  # Updated model name with correct prefix

    def create_embeddings(self, chunks):
        """
        Create text embeddings for a list of text chunks.

        :param chunks: List of string chunks from the document
        :return: List of embeddings corresponding to each chunk
        """
        return [self._get_embedding(chunk) for chunk in chunks]

    def get_question_embedding(self, question):
        """
        Get an embedding for a user's question.

        :param question: String representing the user's question
        :return: Embedding vector for the question
        """
        return self._get_embedding(question)

    def _get_embedding(self, text):
        """
        Helper method to get a single embedding.

        :param text: Text to embed
        :return: Embedding vector
        """
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="semantic_similarity"
        )
        return result['embedding']

    def display_plasma_ball(self, question_point=None, highlighted_indices=None):
        if self.reduced_embeddings is None:
            st.warning("Please upload a PDF first to visualize embeddings.")
            return
        
        # Convert embeddings to numpy array for easier manipulation
        points = np.array(self.reduced_embeddings)
        
        # Rest of the visualization code...

    def run(self):
        st.title("PDF Plasma Viewer")
        
        # Sidebar controls
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
            num_chunks = st.slider("Number of chunks", 20, 100, 20)
            
            if uploaded_file:
                self.process_pdf(uploaded_file, num_chunks)
            
            if self.reduced_embeddings is not None:
                question = st.text_input("Enter your question")
                n_neighbors = st.slider("Number of nearest neighbors", 1, 10, 5)
                
                if question:
                    self.process_question(question, n_neighbors)

        # Main view
        if self.reduced_embeddings is not None:
            self.display_plasma_ball()

# Global instance for use in main.py
embedding_manager = EmbeddingManager()

def create_embeddings(chunks):
    """
    Wrapper for creating embeddings to be used in main.py.

    :param chunks: List of text chunks
    :return: List of embeddings
    """
    return embedding_manager.create_embeddings(chunks)

def get_question_embedding(question):
    """
    Wrapper for getting question embedding to be used in main.py.

    :param question: The question text
    :return: Embedding vector for the question
    """
    return embedding_manager.get_question_embedding(question)