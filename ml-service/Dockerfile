# Build from the project root:
#   docker build -t ml-service -f ml-service/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY ml-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml-service/ /app/
COPY lib/ /app/lib/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
