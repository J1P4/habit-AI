FROM python:3.8.10

ENV PYTHONUNBUFFERED=1
WORKDIR /app


COPY . .

RUN apt-get update

# Python 관련 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

