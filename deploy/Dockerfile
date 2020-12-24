FROM python:3.8-slim as base

FROM base as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN pip install --prefix=/install -r /requirements.txt && \
    rm -r /root/.cache && \
    rm -rf /src

FROM base

COPY --from=builder /install /usr/local
COPY scripts /app

RUN apt-get update -y && \
    apt-get -y install libsndfile1 

WORKDIR /app

EXPOSE 5000

CMD ["python3" ,"server.py"]