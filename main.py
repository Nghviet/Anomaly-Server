import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import association_rules
import itertools
import copy
import numpy as np
from scipy.stats import stats
import random
import pickle
from mlxtend.preprocessing import TransactionEncoder
from random import randrange
import warnings
import sys, os, time, math
import ssl

from threading import Thread
import threading

CONNECTION_STRING = "mongodb://{}:{}@{}".format(os.environ['DB_USER'], os.environ['DB_PASS'], os.environ['DB_URL'])
# CONNECTION_STRING = "mongodb://{}:{}@{}".format("root","rootroot","112.137.129.202:27018")
from pymongo import MongoClient
mongoClient = MongoClient(CONNECTION_STRING)
print(mongoClient)


from miner import process
from paho.mqtt import client as mqtt
client = mqtt.Client()
client.tls_set(ca_certs="ca.crt", certfile="mqtt.crt", keyfile="mqtt.key",tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)
client.tls_insecure_set(True)
client.connect(os.environ['MQTT_URL'], int(os.environ['MQTT_PORT']), keepalive = 60) #connect to broker
# client.connect("112.137.129.202", 8883, keepalive = 60) #connect to broker

def on_message(client, userdata, message):
	print("Received message '" + str(message.payload) + "' on topic '"
	    + message.topic + "' with QoS " + str(message.qos))
	split = message.topic.split("/")
	print("{}/{}".format(split[0],split[1]))
	t = threading.Thread(target=thread, args=("{}/{}".format(split[0],split[1]),))
	t.start()


def on_connect(client, userdata, flags, rc):
	print("MQTT connected")
	client.subscribe("+/+/request/mining")

def on_disconnect(client, userdata, rc):
	print("MQTT disconnected {}".format(rc))

def thread(cn):
	print("Start mining for {}".format(cn))
	process(mongoClient,cn)
	client.publish("{}/response/mining".format(cn))


client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.loop_forever()