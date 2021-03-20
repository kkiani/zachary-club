FROM pytorch/pytorch:latest

RUN pip install --no-cache dgl numpy fastapi uvicorn[standard]


COPY train.py /usr/bin/train
COPY model /usr/bin/model
COPY serve.py /usr/bin/serve

RUN chmod 755 /usr/bin/train
RUN chmod 755 /usr/bin/serve

EXPOSE 8080