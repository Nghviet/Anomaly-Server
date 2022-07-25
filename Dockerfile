FROM python:3.8.13-slim

WORKDIR /vietnh/

COPY requirement.txt requirement.txt
RUN pip install -r requirement.txt

COPY ca.crt ca.crt
COPY mqtt.crt mqtt.crt
COPY mqtt.key mqtt.key
COPY main.py main.py
COPY miner.py miner.py

CMD python3 main.py