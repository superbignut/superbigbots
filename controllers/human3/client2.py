
import math
import multiprocessing
import time
from multiprocessing.connection import Listener, Connection, PipeConnection, Client

while True:
    try:
        client = Client(('127.0.0.1', 8004)) # 向目标地址发起一个连接
        break
    except:
        print("wait")
        time.sleep(1)
        
        continue

while True:

    data = client.recv()
    print(data)