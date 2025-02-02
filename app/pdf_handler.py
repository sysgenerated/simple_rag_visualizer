import fitz  # PyMuPDF
import re

class PDFHandler:
    def __init__(self):
        self.chunks = []
        self.default_chunk_size = 250
        
    def estimate_tokens(self, text):
        """
        Rough estimation of tokens in text (approximates GPT-style tokenization).
        
        :param text: Input text
        :return: Estimated token count
        """
        # This is a simple approximation - about 4 characters per token
        return len(text) // 4

    def process_pdf(self, pdf_content, chunk_size=None):
        """
        Process PDF content with layout-aware chunking.
        
        :param pdf_content: Bytes of PDF content
        :param chunk_size: Target number of tokens per chunk (default: 250)
        :return: List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size
            
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        # Open PDF from memory
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            # Get blocks with layout information
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # block[4] contains the text content
                text = block[4].strip()
                if not text:
                    continue
                    
                # Split block into sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_tokens = self.estimate_tokens(sentence)
                    
                    # If a single sentence exceeds max size, split it into smaller parts
                    if sentence_tokens > 500:
                        words = sentence.split()
                        temp_chunk = []
                        temp_tokens = 0
                        
                        for word in words:
                            word_tokens = self.estimate_tokens(word)
                            if temp_tokens + word_tokens > chunk_size:
                                if temp_chunk:
                                    chunks.append(" ".join(temp_chunk))
                                temp_chunk = [word]
                                temp_tokens = word_tokens
                            else:
                                temp_chunk.append(word)
                                temp_tokens += word_tokens
                        
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        continue
                    
                    # If adding this sentence would exceed target size, start new chunk
                    if current_token_count + sentence_tokens > chunk_size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_token_count = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_token_count += sentence_tokens
            
            # End of page - if we have content, add as chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
        
        doc.close()
        self.chunks = chunks
        return chunks

# Global instance for use in main.py
pdf_handler = PDFHandler()

def process_pdf(pdf_content, chunk_size=None):
    """
    Wrapper for PDF processing to be used in main.py.
    
    :param pdf_content: Byte content of the PDF
    :param chunk_size: Target number of tokens per chunk
    :return: List of text chunks
    """
    return pdf_handler.process_pdf(pdf_content, chunk_size)

def get_chunk_text(index):
    """
    Get the text content of a specific chunk.
    
    :param index: Index of the chunk
    :return: Text content of the chunk
    """
    if 0 <= index < len(pdf_handler.chunks):
        return pdf_handler.chunks[index]
    raise IndexError("Chunk index out of range")