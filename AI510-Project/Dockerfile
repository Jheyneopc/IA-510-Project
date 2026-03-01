FROM python:3.10-slim

WORKDIR /app

COPY model/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV ARTIFACTS_DIR=model/artifacts
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]