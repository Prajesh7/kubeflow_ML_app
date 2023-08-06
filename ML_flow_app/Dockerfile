FROM python:3.7-slim

RUN sudo apt update -y && sudo apt install awscli -y
WORKDIR /app

COPY . /app
RUN sudo pip install -r requirements.txt

CMD ["python3", "app.py"]