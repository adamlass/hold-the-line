

from queue import Queue
from threading import Thread
import time


q = Queue()


def test():
    print("Thread started, getting text", flush=True)
    text = q.get()
    print(text, flush=True)
    
# run test on thread

thread = Thread(target=test)

thread.start()

time.sleep(2)


q.put("Hello World")

thread.join()