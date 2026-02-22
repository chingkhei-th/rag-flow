import sys
import threading
import itertools
import time
from src.rag_pipeline import RAGPipeline

class Spinner:
    def __init__(self, message="Thinking"):
        self.message = message
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.stop_running = threading.Event()
        self.spinner_thread = None

    def start(self):
        self.stop_running.clear()
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    def spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(f'\r{self.message} {next(self.spinner_chars)}')
            sys.stdout.flush()
            time.sleep(0.1)

    def stop(self):
        self.stop_running.set()
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 4) + '\r')
        sys.stdout.flush()
def main():
    print("Initializing Advanced Context-Aware RAG Pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("\nStarting document ingestion process...")
    pipeline.ingest_documents()

    print("\n==================================")
    print("RAG System Ready. Type 'exit' to quit.")
    print("==================================\n")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            if not user_input.strip():
                continue

            spinner = Spinner("Thinking")
            spinner.start()

            try:
                response = pipeline.query_stream(user_input)
                answer_stream = response["answer_stream"]

                first_chunk = next(answer_stream, None)
            finally:
                spinner.stop()

            print("\nAI: ", end="", flush=True)
            if first_chunk:
                print(first_chunk, end="", flush=True)
            for chunk in answer_stream:
                print(chunk, end="", flush=True)
            print()

            print("\nSources:")
            if "context" in response and response["context"]:
                seen_sources = set()
                for i, doc in enumerate(response["context"]):
                    source = doc.metadata.get("source", "Unknown source")
                    page = doc.metadata.get("page", "N/A")
                    key = f"{source}_page_{page}"
                    if key not in seen_sources:
                        print(f" - {source} (Page: {page})")
                        seen_sources.add(key)
            else:
                print(" No context matched.")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}\n")

if __name__ == "__main__":
    main()
