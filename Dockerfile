FROM python:3.9-slim
   
# Add ARG for build-time variable
ARG GOOGLE_API_KEY

# Set as environment variable
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Create .streamlit directory and copy config
COPY .streamlit /app/.streamlit

# Copy application code
COPY app/ .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
