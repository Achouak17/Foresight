# 1. Base Image
FROM python:3.9-slim-bullseye

# 2. Install Java 11 + libgomp + essential system tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        libgomp1 \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 4. App directory
WORKDIR /app

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy everything
COPY . .

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Start Streamlit
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
