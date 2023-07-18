FROM python:3.7-slim

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p app

COPY ./app app

EXPOSE 80

CMD ["uvicoren", "app.main:app", "host", "127.0.0.1", "--port", "80"]