import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue


class AsyncQueueProcessor:
    def __init__(self, num_workers, callback, max_queue_size=50):
        self.num_workers = num_workers
        self.callback = callback
        self.data_queue = Queue(maxsize=max_queue_size)
        self.results = []
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.running = threading.Event()
        self.running.set()

        # Start worker threads
        self.futures = [self.executor.submit(self.worker) for _ in range(num_workers)]

    def add_data(self, data):
        """Put data into the queue to be processed."""
        self.data_queue.put(data)

    def worker(self):
        """Worker function that processes items in the queue."""
        while self.running.is_set() or not self.data_queue.empty():
            try:
                data = self.data_queue.get(timeout=1)  # Timeout to check running state
                result = self.callback(data)
                self.results.append(result)
                self.data_queue.task_done()
            except Empty:
                continue  # Continue if there's no data to process yet

    def stop(self):
        """Stop all workers gracefully."""
        self.running.clear()
        self.executor.shutdown(wait=True)

    def get_results(self):
        """Return current processed results."""
        return self.results

    def is_running(self):
        """Check if the processor is still running."""
        return self.running.is_set()

    def join(self):
        while not self.data_queue.empty():
            time.sleep(1)
        self.stop()


# Example usage
if __name__ == "__main__":
    # Callback function to process data
    def process_data(data):
        time.sleep(1)  # Simulate processing time
        return f"Processed {data}"

    # Create an instance with 3 workers and the callback function
    processor = AsyncQueueProcessor(num_workers=3, callback=process_data)

    # Add data to the queue
    for i in range(10):
        print(f"Adding data {i}")
        processor.add_data(f"Data {i}")

    # Allow some time to process
    print("Waiting for processing...")
    time.sleep(5)

    # Stop the processor and print results
    print("Stopping processor...")
    processor.stop()
    print("Results:", processor.get_results())
