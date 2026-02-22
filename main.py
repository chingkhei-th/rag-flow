import sys
from src.rag_pipeline import RAGPipeline

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

            print("Thinking...")
            response = pipeline.query(user_input)

            print(f"\nAI: {response['answer']}")
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
