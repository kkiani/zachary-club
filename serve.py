#!/usr/bin/env python

import os
from pandas.core.algorithms import mode
from starlette.routing import Host
import uvicorn
from fastapi import FastAPI, Response
from io import StringIO
import pandas as pd
import torch as th


app = FastAPI()
model_dir = '/opt/ml/model'
model = th.load(os.path.join(model_dir, 'karate_club.pt'))


app.get('/ping')
def ping():
    return Response(content='\n', status_code=200)

app.post('/invocations')
def predict(data: bytes):
    data_decoded = data.decode('utf-8')
    data_stream = StringIO(data_decoded)
    df = pd.read_csv(data_stream, header=None)
    response = model.predict(df)

    return Response(content=str(response), status_code=200)


def main():
    uvicorn.run(app, port=8080, host='0.0.0.0')

if __name__ == '__main__':
    main()