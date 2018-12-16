import os
from random import random

from google.oauth2 import service_account
from googleapiclient import discovery

from trainer import model


PATH_CREDENTIALS_GOOGLE = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None)

credentials = service_account.Credentials.from_service_account_file(PATH_CREDENTIALS_GOOGLE)
api = discovery.build('ml', 'v1', credentials=credentials)

parent = 'projects/%s/models/%s/versions/%s' % ('academic-motif-193414', 'neural_keras_model', 'v1')
data = model.generator_input(batch_size=1)

for i in range(0, 30):
    flag = True
    while flag:
        sample = next(data)
        flag = True if random() > 0.5 else False

    sample = next(data)
    input = sample[0][0].tolist()
    request_data = {'instances': [input]}
    response = api.projects().predict(body=request_data, name=parent).execute()
    result = response['predictions'][0]['income'][0]
    print("{:.3g} value predicted for the sample {}, binary output {}".format(result, input, round(result)))
