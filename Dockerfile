FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY cloudops_env/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Set Python path to ensure cloudops_env is importable as a package
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 7860

# Health check (OpenEnv requirement)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "cloudops_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
