FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY models/ models/
# Dataset is strictly not copied into the Docker container for security/compliance.
# The container is meant for running the API and Dashboard, not retraining on raw PHI data.

EXPOSE 8000
EXPOSE 8501

# By default, run the API. (Streamlit could be run via a separate entrypoint/compose)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
