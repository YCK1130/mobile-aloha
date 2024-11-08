import queue
import select
import sys
import threading
import time


class CommandLineMonitor:

    def __init__(self, debounce_time=0.5):
        self.running = False
        # Flag to control input processing based on parameter changes
        self.input_text = ""
        self.input_complete = False

        self.__input_queue = queue.Queue()  # Queue to manage user input
        self.debounce_time = debounce_time  # Debounce time in seconds

        self.input_thread = threading.Thread(target=self.__monitor_user_input)

    def __monitor_user_input(self):
        """Non-blocking monitoring of user input with debouncing."""
        buffer = ""  # Buffer to collect input characters
        last_input_time = None

        self.__clear_input()  # Clear any pending input before starting
        while self.running:
            # Non-blocking check for user input
            if (
                sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]
            ):  # Timeout for quick exit
                buffer += sys.stdin.read(1)
                last_input_time = time.time()  # Update last input time for debounce

            # Process complete input if debounce time has passed since the last input
            if (
                last_input_time and time.time() - last_input_time >= self.debounce_time
            ) or buffer.endswith("\n"):
                if buffer.strip():
                    self.__input_queue.put(buffer.strip())
                if buffer.endswith("\n"):
                    self.__flush_input()
                buffer = ""  # Clear buffer after processing
            time.sleep(0.1)  # Small delay to avoid busy waiting

    def __clear_input(self):
        """Clear all pending input in the queue."""
        while not self.__input_queue.empty():
            try:
                self.__input_queue.get_nowait()  # Discard input from the queue
            except queue.Empty:
                break

    def __flush_input(self):
        """Flush the input buffer."""
        self.input_text = ""
        self.input_complete = False

        while not self.__input_queue.empty():
            try:
                self.input_text += self.__input_queue.get_nowait()
            except queue.Empty:
                break
        self.input_complete = True

    def get_input(self):
        """Get the complete user input string."""
        return self.input_text, self.input_complete and self.__input_queue.empty()

    def stop(self):
        """Stop all monitoring threads gracefully."""
        self.running = False
        self.input_thread.join()
        print("Monitoring stopped.")

    def run(self):
        self.running = True
        self.input_thread.start()

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


# Example usage
if __name__ == "__main__":
    with CommandLineMonitor() as monitor:
        print("Monitoring started. Type 'exit' to stop.")

    # The program will continue running until 'exit' is typed in the command line.
