# !/usr/bin/env python
import pika

class Sender:
    def sendMessage(self,message):
        credentials = pika.PlainCredentials('admin', 'admin')
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            '127.0.0.1', 5672, '/', credentials))
        channel = connection.channel()
        channel.queue_declare(queue='balance', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='balance',
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,
                # make message persistent
            )
        )
        connection.close()
