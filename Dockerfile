FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY fetch_dataset.py ./fetch_dataset.py
COPY train.py ./train.py

ENV PYTHONPATH=/app

RUN python fetch_dataset.py && python train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
