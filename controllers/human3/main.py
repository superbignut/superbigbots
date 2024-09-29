
import math
import multiprocessing
import time
from multiprocessing.connection import Listener, Connection, PipeConnection

def do_socket(conn:Connection|PipeConnection, addr):
    try:
        while True:
            if conn.poll(timeout=1) == False: # 连接是否有数据
                time.sleep(0.5)
                continue
            data = conn.recv()
            conn.send("111")
            print(data)
    except Exception as e:
        print(e)
    finally:
        conn.close()
        

def run_server(host, port):
    server_sock = Listener((host, port))
    pool = multiprocessing.Pool(10)
    while True:
        conn = server_sock.accept()
        addr = server_sock.last_accepted
        print(addr)

        pool.apply_async(func=do_socket, args=(conn, addr,))  # 每当有一个连接过来调用， 不等待结果

if __name__ == '__main__':
    server_host = '127.0.0.1'
    server_port = 8000
    run_server(server_host, server_host)
