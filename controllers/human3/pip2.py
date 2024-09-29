import multiprocessing
from multiprocessing import shared_memory
import time

existing_shm = shared_memory.SharedMemory(name='cmdcmd')
print(bytes(existing_shm.buf[:7]).decode('utf8'))