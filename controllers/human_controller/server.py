"""
    最开始的想法是把狗和人一个当作服务器一个当作客户端进行tcp通信，但是试了一下会报错，就只能都当作客户端，

    所有需要另外开启一个中继服务器，来进行消息的中转

"""
import math
import multiprocessing
import time
from multiprocessing.connection import Listener, Connection, PipeConnection
import select
def do_socket(conn1:Connection, conn2:Connection):
    # 收到一个连接的数据则向另一个连接转发
    try:
        while True:
            if conn1.poll(timeout=1) == False: # 连接是否有数据
                
                if conn2.poll(timeout=1) == False:
                    time.sleep(1)
                else:
                    data = conn2.recv()
                    conn1.send(data)
                    continue
                continue
            data = conn1.recv()
            conn2.send(data)
    except Exception as e:
        print(e)
    finally:
        conn1.close()
        conn2.close()
        

def run_server(host, port):
    server_sock = Listener((host, port))
    while True:
        conn1 = server_sock.accept()
        print(conn1)
        conn2 = server_sock.accept()
        print(conn2)
        do_socket(conn1, conn2)


if __name__ == '__main__':
    server_host = '127.0.0.2'
    server_port = 8004
    run_server(server_host, server_port)
