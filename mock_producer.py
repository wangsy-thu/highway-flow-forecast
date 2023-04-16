import json

import pika

from utils.protocol import byte2array, array2byte

user_auth = pika.PlainCredentials(
    username='WangY',
    password='WangY@20010418@WangY'
)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='101.42.231.27',
        credentials=user_auth
    )
)

channel = connection.channel()
channel.queue_declare('flow-data-channel')

if __name__ == '__main__':

    flow_dict = {}

    for i in range(207):
        flow_dict[str(i)] = [
            1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
        ]

    channel.basic_publish(
        exchange='',
        routing_key='flow-data-channel',
        body=json.dumps(flow_dict).encode()
    )
    print('=====Send Success=====')
