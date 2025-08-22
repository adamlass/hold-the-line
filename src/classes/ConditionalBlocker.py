
from threading import Lock

class ConditionalBlocker():
    lock: Lock
    logger = None
    
    def __init__(self, lock:Lock, blocking: bool = False, logger=None):
        self.lock = lock
        self.blocking = blocking
        self.logger = logger
    
    def log(self, *args, flush: bool=True, end: str="\n"):
        if self.logger is not None:
            self.logger(*args, flush=flush, end=end)
        else:
            print(*args, flush=flush, end=end)  
            
    def __enter__(self):
        if self.blocking:
            # self.log("Acquiring lock")
            self.lock.acquire()
            # self.log("Lock acquired")
            
        return self
    
    def __exit__(self, type, value, traceback):
        if self.blocking:
            # self.log("Releasing lock")
            self.lock.release()
            # self.log("Lock released")
            
        if type is not None:
            # self.log("Error occurred:", type, value)
            raise value
            
        return self