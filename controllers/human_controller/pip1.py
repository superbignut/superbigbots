import multiprocessing
import time
from multiprocessing import shared_memory
# 创建一个共享的整型变量
try:
    shm_a = shared_memory.SharedMemory(name = 'cmdcmd', create=True, size=10)
except FileExistsError as e:
    existing_shm = shared_memory.SharedMemory(name='cmdcmd')
    existing_shm.close()
    existing_shm.unlink()
    time.sleep(3)
    shm_a = shared_memory.SharedMemory(name = 'cmdcmd', create=True, size=10)