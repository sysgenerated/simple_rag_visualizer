# simple_rag_visualizer

Create code that does the following.

1. Create a streamlit application
2. Allows a user to upload a PDF
3. Uses PyMUPDF to perform layout aware chunking. The chunks should be between 100 and 500 tokens. Users should be able to select a different chunk size. The default should be 250 tokens.
4. Creates embeddings for the chunks using Google AI Studio
5. Uses UMAP to reduce the embedding dimensionality to unit vectors in 3d
6. Displays the embeddings as points on a ball. The ball should be look semi transparent and the embeddings should be lines flowing from the center and touching the ball. This should look similar to an 80s and 90s plasma ball.
7. Allow a user to enter a question
8. Create an embedding from the question and then use the same UMAP object to reduce dimensionality to unit vectors in 3d.
9. Display the question embedding on the same ball but highlighted differently so it's noticeable
10. Allow the user to select a number of nearest neighbors and highlight the nearest neighbors on the ball
11. Display in some way the text associated with a given embedding point when it is highlighted or clicked.
12. Allow the user to rotate the ball
13. All of the code should be able to reside in a single docker container