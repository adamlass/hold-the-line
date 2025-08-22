from abc import ABC, abstractmethod
from multiprocessing.context import ForkServerProcess
import queue
from threading import Thread
from typing import List

from classes.ConditionalBlocker import ConditionalBlocker
from classes.Environment import Environment
from colorama import Fore, Style
from queue import Queue, Empty, Full

    
class Transceiver(ABC):
    input_queue: Queue
    output_queues: List[Queue]
    wait_for_input: bool
    wait_for_output: bool
    name: str
    environment: Environment
    log_color: str
    thread: Thread
    running: bool
    blocker: ConditionalBlocker 
    subscribed_to: List["Transceiver"]
    batch_processing: bool
    buffer_size: int
    
    def __init__(self, name: str, environment: Environment, log_color: str = Fore.WHITE, wait_for_input: bool=False, blocking: bool = False, batch_processing: bool = False, buffer_size: int = -1, wait_for_output: bool = False, debug: bool = False):
        self.name = name
        self.environment = environment
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queues = []
        self.wait_for_input = wait_for_input
        self.wait_for_output = wait_for_output
        self.log_color = log_color
        self.running = False
        self.blocker = ConditionalBlocker(self.environment.lock, blocking, self.log)
        self.subscribed_to = []
        self.batch_processing = batch_processing
        self.buffer_size = buffer_size
        self.debug = debug
        
    def subscribe(self, transceiver: "Transceiver"):
        if transceiver.input_queue is None:
            self.log("Transceiver doesn't have an active input queue. Will not subscribe.")
            return
        self.output_queues.append(transceiver.input_queue)
        transceiver.subscribed_to.append(self)
        
    def unsubscribe(self, transceiver: "Transceiver"):
        for queue in self.output_queues:
            if queue == transceiver.input_queue:
                self.output_queues.remove(queue)
                break
        
    def transmit(self, data: any):
        for queue in self.output_queues:
            queue.put(data, block=self.wait_for_output)
                
    def _receive(self, timeout: float | None = None) -> any:
        try:
            return self.input_queue.get(block=self.wait_for_input, timeout=timeout)
        except Empty as e:
            if self.wait_for_input:
                self.log(f"No data received after {timeout} seconds:", e)
            return None
        except Full as e:
            if self.wait_for_input:
                self.log(f"Input queue is full. Will not receive data:", e)
            return None
        
    def __batch_receive(self, timeout: float | None = None) -> List[any]:
        batch = []
        
        while len(batch) == 0 or self.input_queue.qsize() > 0:
            data = self._receive(timeout)
            if data is None:
                break
            batch.append(data)
        
        if len(batch) == 0:
            return None
        
        return batch
    
    def log(self, *args, flush: bool=True, end: str="\n"):
        print(self.environment.log_color, 
              f"[{self.environment.id}]",
              Style.RESET_ALL,
              f"{self.log_color}{self.name}:",
              *args,
              Style.RESET_ALL,
              flush=flush,
              end=end)
        
    def run(self) -> Thread:
        self.setup()
        self.running = True
        self.thread = Thread(target=self._loop)
        self.thread.start()
        if self.debug: self.log("Started")
        return self.thread
    
    def join(self):
        self.running = False
        for transceiver in self.subscribed_to:
            if self.debug: self.log("Unsubscribing from transceiver:", transceiver.name)
            transceiver.unsubscribe(self)
        
        if self.debug: self.log("Waiting for output queues to join")
        for queue in self.output_queues:
            queue.join()
        if self.debug: self.log("Output queues joined")
        
        if self.wait_for_input:
            self.input_queue.put(None)
            
        self.teardown()
        self.thread.join()
        if self.debug: self.log("Stopped")
        
    def _loop(self):
        while self.running:
            try:
                if self.batch_processing:
                    data = self.__batch_receive()
                    length = len(data)
                else:
                    data = self._receive()
                    length = 1
                    
                with self.blocker:
                    output = self.process(data)
                    
                self._processing_done(length)
                    
                if output is not None:
                    self.transmit(output)
                    
            except Exception as e:
                self.log("Error processing data:", e)
        if self.debug: self.log("Exiting loop")

    def _processing_done(self, n: int = 1):
        for _ in range(n):
            self.input_queue.task_done()

    @abstractmethod
    def process(self, data: any) -> any:
        """Returning None will not transmit any data."""
        pass
    
    @abstractmethod
    def setup(self):
        pass
    
    @abstractmethod
    def teardown(self):
        pass
        