import streamlit as st
import plotly.graph_objects as go
from pdf_handler import PDFHandler
from embedding_manager import EmbeddingManager
from umap_visualizer import UMAPVisualizer
import numpy as np
import os
from dotenv import load_dotenv
import time

# Load environment variables at startup
load_dotenv()

class PlasmaApp:
    def __init__(self):
        self.pdf_handler = PDFHandler()
        self.embedding_manager = EmbeddingManager()
        self.umap_visualizer = UMAPVisualizer()
        self.reduced_embeddings = None
        self.chunks = None
        self.embeddings = None

    def run(self):
        # Set page config first, before any other Streamlit commands
        st.set_page_config(layout="wide")
        
        st.title("PDF Plasma Viewer")
        
        # Initialize session state
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'question_point' not in st.session_state:
            st.session_state.question_point = None
        if 'highlighted_indices' not in st.session_state:
            st.session_state.highlighted_indices = None
        if 'question' not in st.session_state:
            st.session_state.question = ""
        if 'n_neighbors' not in st.session_state:
            st.session_state.n_neighbors = 5
        if 'visualization_key' not in st.session_state:
            st.session_state.visualization_key = 0
        if 'embeddings' not in st.session_state:
            st.session_state.embeddings = None
        if 'umap_model' not in st.session_state:
            st.session_state.umap_model = None
            
        def handle_question_submit():
            if st.session_state.question:
                with st.spinner("Processing question..."):
                    # Get question embedding
                    question_embedding = self.embedding_manager.get_question_embedding(st.session_state.question)
                    if question_embedding is None:
                        st.error("Failed to generate question embedding")
                        return
                        
                    # Restore UMAP model
                    if st.session_state.umap_model is None:
                        st.error("UMAP model not found")
                        return
                    self.umap_visualizer.umap_reducer = st.session_state.umap_model
                    
                    try:
                        # Transform question embedding
                        question_3d = self.umap_visualizer.transform_new_embedding(question_embedding)
                        if question_3d is None:
                            st.error("Failed to transform question embedding")
                            return
                            
                        # Find nearest neighbors
                        highlighted_indices = self.umap_visualizer.get_nearest_neighbors(
                            question_3d, 
                            st.session_state.n_neighbors
                        )
                        
                        # Update session state
                        st.session_state.question_point = question_3d
                        st.session_state.highlighted_indices = highlighted_indices
                        st.session_state.visualization_key += 1
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        return
        
        # Create sidebar for controls
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
            
            # Information about chunk size
            st.info("Documents are automatically split into chunks of approximately 250 tokens each.")
            
            # Only process PDF if file is uploaded and hasn't been processed yet
            if uploaded_file and not st.session_state.pdf_processed:
                with st.spinner("Processing PDF..."):
                    self.process_pdf(uploaded_file, chunk_size=250)
                    if self.reduced_embeddings is not None:
                        st.session_state.embeddings = self.reduced_embeddings
                        st.session_state.umap_model = self.umap_visualizer.umap_reducer
                        st.session_state.pdf_processed = True
                    else:
                        st.error("Failed to process PDF")
            
            # Show question input if we have processed embeddings
            if st.session_state.pdf_processed:
                st.text_input(
                    "Enter your question",
                    key="question"
                )
                
                st.slider(
                    "Number of nearest neighbors",
                    1, 10, 5,
                    key="n_neighbors"
                )
                
                st.button("Process Question", on_click=handle_question_submit)
                
                # Show relevant chunks if we have highlighted indices
                if st.session_state.highlighted_indices is not None:
                    st.subheader("Most Relevant Chunks:")
                    for idx in st.session_state.highlighted_indices:
                        with st.expander(f"Chunk {idx + 1}"):
                            st.write(self.chunks[idx])

        # Main content area - full width minus sidebar
        if st.session_state.pdf_processed:
            self.reduced_embeddings = st.session_state.embeddings
            self.umap_visualizer.umap_reducer = st.session_state.umap_model
            self.display_plasma_ball(
                question_point=st.session_state.question_point,
                highlighted_indices=st.session_state.highlighted_indices,
                key=f"plasma_ball_{st.session_state.visualization_key}"
            )
        else:
            st.info("Upload a PDF to begin visualization.")

    def process_pdf(self, uploaded_file, chunk_size):
        """Process PDF and create embeddings"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Process PDF into chunks
        status_text.text("Chunking PDF into segments...")
        pdf_content = uploaded_file.read()
        self.chunks = self.pdf_handler.process_pdf(pdf_content, chunk_size)
        progress_bar.progress(25)
        st.write(f"Created {len(self.chunks)} chunks from PDF")
        
        # Step 2: Create embeddings
        status_text.text("Creating embeddings for chunks...")
        self.embeddings = self.embedding_manager.create_embeddings(self.chunks)
        progress_bar.progress(50)
        st.write(f"Generated embeddings of dimension {len(self.embeddings[0])}")
        
        # Step 3: Reduce dimensions with UMAP
        status_text.text("Reducing dimensionality with UMAP...")
        self.reduced_embeddings = self.umap_visualizer.reduce_dimensions(self.embeddings)
        progress_bar.progress(75)
        
        # Step 4: Prepare nearest neighbors model
        status_text.text("Preparing nearest neighbors model...")
        self.umap_visualizer.prepare_nearest_neighbors()
        progress_bar.progress(100)
        
        status_text.text("PDF processing complete!")
        time.sleep(1)  # Give users a moment to see the completion message
        status_text.empty()
        progress_bar.empty()

    def display_plasma_ball(self, question_point=None, highlighted_indices=None, key=None):
        if self.reduced_embeddings is None:
            st.info("Upload a PDF to begin visualization.")
            return
            
        # Convert embeddings to numpy array for easier manipulation
        points = np.array(self.reduced_embeddings)
        
        # Create figure
        fig = go.Figure()
        
        # Add semi-transparent sphere with wireframe for better depth perception
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(100), np.cos(v))
        
        # Add wireframe sphere
        fig.add_surface(
            x=x, y=y, z=z,
            opacity=0.1,
            showscale=False,
            hoverinfo='skip',
            surfacecolor=np.ones_like(x),
            colorscale=[[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']],
            contours=dict(
                x=dict(highlight=False),
                y=dict(highlight=False),
                z=dict(highlight=False)
            )
        )
        
        # Add lines from center to points with hover text
        for i, point in enumerate(points):
            color = 'rgba(100,149,237,0.6)'  # Cornflower blue with transparency
            width = 2
            
            if highlighted_indices and i in highlighted_indices:
                color = 'rgba(255,0,0,0.8)'  # Red with less transparency
                width = 4
            
            # Add small sphere at endpoint
            fig.add_scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                ),
                hovertext=f"Chunk {i+1}",
                hoverinfo='text',
                showlegend=False
            )
            
            # Add line from center
            fig.add_scatter3d(
                x=[0, point[0]], 
                y=[0, point[1]], 
                z=[0, point[2]],
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='skip',
                showlegend=False
            )
        
        # Add question point if exists
        if question_point is not None:
            # Add endpoint sphere for question
            fig.add_scatter3d(
                x=[question_point[0]],
                y=[question_point[1]],
                z=[question_point[2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(0,255,0,1)',
                    symbol='diamond'
                ),
                hovertext="Question",
                hoverinfo='text',
                showlegend=False
            )
            
            # Add line from center for question
            fig.add_scatter3d(
                x=[0, question_point[0]], 
                y=[0, question_point[1]], 
                z=[0, question_point[2]],
                mode='lines',
                line=dict(color='rgba(0,255,0,0.8)', width=6),
                hoverinfo='skip',
                showlegend=False
            )
        
        # Update layout for better visualization
        fig.update_layout(
            scene = dict(
                xaxis=dict(range=[-1.2, 1.2], showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(range=[-1.2, 1.2], showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(range=[-1.2, 1.2], showticklabels=False, showgrid=False, zeroline=False),
                aspectmode='cube',  # Force equal aspect ratio
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),  # Adjust default camera position
                ),
                dragmode='orbit'  # Make rotation more intuitive
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hovermode='closest'
        )
        
        # Add help text
        st.markdown("""
        **Visualization Guide:**
        - 🔵 Blue lines: Document chunks
        - 🔴 Red lines: Similar chunks to question
        - 💚 Green line: Question vector
        - 🖱️ Drag to rotate, scroll to zoom
        - Hover over endpoints to see chunk numbers
        """)
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app = PlasmaApp()
    app.run()