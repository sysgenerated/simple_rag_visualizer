import umap
import numpy as np
from sklearn.neighbors import NearestNeighbors

class UMAPVisualizer:
    def __init__(self):
        # UMAP instance for dimensionality reduction
        self.umap_reducer = umap.UMAP(n_components=3, random_state=42)
        # We'll store the embeddings here after reducing dimensions for later use
        self.reduced_embeddings = None
        # Nearest Neighbors model for finding neighbors
        self.nn_model = None

    def reduce_dimensions(self, embeddings):
        """
        Reduce the dimensionality of embeddings to 3D unit vectors.
        
        :param embeddings: List of embedding vectors
        :return: List of reduced 3D vectors
        """
        self.reduced_embeddings = self.umap_reducer.fit_transform(embeddings)
        
        # Normalize vectors to unit vectors
        magnitudes = np.linalg.norm(self.reduced_embeddings, axis=1)
        self.reduced_embeddings = self.reduced_embeddings / magnitudes[:, np.newaxis]
        
        return self.reduced_embeddings.tolist()

    def prepare_nearest_neighbors(self):
        """
        Prepare the NearestNeighbors model for later use.
        """
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(self.reduced_embeddings)

    def get_nearest_neighbors(self, question_vector, n_neighbors=5):
        """
        Find the nearest neighbors for a given question vector.

        :param question_vector: The vector of the question in reduced 3D space
        :param n_neighbors: Number of neighbors to return
        :return: Indices of nearest neighbors
        """
        if self.nn_model is None:
            self.prepare_nearest_neighbors()
        
        distances, indices = self.nn_model.kneighbors([question_vector], n_neighbors=n_neighbors)
        return indices[0].tolist()

    def transform_question(self, question_embedding):
        """
        Transform a question embedding using the fitted UMAP model.
        
        :param question_embedding: The embedding vector for the question
        :return: Reduced 3D vector
        """
        reduced = self.umap_reducer.transform([question_embedding])
        # Normalize to unit vector
        magnitude = np.linalg.norm(reduced)
        return (reduced / magnitude)[0]

    def transform_new_embedding(self, embedding):
        """
        Transform a new embedding using the existing UMAP transformation.
        
        :param embedding: New embedding vector to transform
        :return: Reduced 3D vector
        """
        if self.umap_reducer is None:
            raise ValueError("UMAP reducer not fitted. Process a PDF first.")
        
        # Ensure embedding is 2D array with shape (1, n_features)
        embedding_array = np.array(embedding).reshape(1, -1)
        
        # Transform using existing UMAP
        reduced = self.umap_reducer.transform(embedding_array)
        
        # Normalize to unit vector
        reduced = reduced / np.linalg.norm(reduced)
        
        return reduced[0]  # Return the single transformed vector

umap_visualizer = UMAPVisualizer()

def reduce_dimensions(embeddings):
    """
    Wrapper for reducing dimensions to be used in main.py.
    
    :param embeddings: List of embedding vectors
    :return: List of reduced 3D vectors
    """
    return umap_visualizer.reduce_dimensions(embeddings)

def get_nearest_neighbors(embedding, n_neighbors=5):
    """
    Wrapper for getting nearest neighbors to be used in main.py.

    :param embedding: The vector of the question in reduced 3D space
    :param n_neighbors: Number of neighbors to return
    :return: List of indices for nearest neighbors
    """
    return umap_visualizer.get_nearest_neighbors(embedding, n_neighbors)