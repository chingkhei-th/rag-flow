import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.embeddings import OllamaRobustEmbeddings

from src.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    CHROMA_DB_DIR,
)

class RAGPipeline:
    def __init__(self):
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Initialize Embeddings using the robust wrapper (avoids NaN on long texts)
        self.embeddings = OllamaRobustEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        # Initialize Vector Store
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )

        # Initialize LLM
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

        # Initialize Text Splitter for Context Chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Create chains
        self._create_chains()

    def _create_chains(self):
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context.
If the context does not contain enough information, say "I don't have enough information to answer this question."

<context>
{context}
</context>

Question: {input}""")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # Format retrieved docs into a plain string for the prompt
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL chain: retrieve -> format -> prompt -> LLM -> parse
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(self.retriever.invoke(x["input"]))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ingest_documents(self):
        # We try loading PDF files, standard approach
        try:
            loader = PyPDFDirectoryLoader(DATA_DIR)
            documents = loader.load()
        except BaseException as e:
            print(f"Warning: Could not load pdfs - {e}")
            documents = []

        # Load .txt files
        try:
            loader_txt = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(loader_txt.load())
        except BaseException as e:
            print(f"Warning: Could not load txt - {e}")

        # Load .md (Markdown) files
        try:
            loader_md = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
            documents.extend(loader_md.load())
        except BaseException as e:
            print(f"Warning: Could not load markdown - {e}")

        existing_items = self.vectorstore.get(include=["metadatas"])

        # Calculate distinct existing documents from metadatas
        existing_sources = set()
        if existing_items and "metadatas" in existing_items and existing_items["metadatas"]:
            for metadata in existing_items["metadatas"]:
                if isinstance(metadata, dict) and "source" in metadata:
                    existing_sources.add(metadata["source"])

        print(f"Number of existing documents in DB: {len(existing_sources)}")

        new_docs = [doc for doc in documents if doc.metadata.get("source") not in existing_sources]

        if new_docs:
            from collections import defaultdict
            doc_groups = defaultdict(list)

            for doc in new_docs:
                source = doc.metadata.get("source", "unknown")
                doc_groups[source].append(doc)

            print(f"Adding/Seeding new documents (Found {len(doc_groups)} new files)")

            for source, docs in doc_groups.items():
                chunks = self.text_splitter.split_documents(docs)
                doc_name = os.path.basename(source)
                print(f"  - Document: {doc_name} | Number of chunks: {len(chunks)}")

                # Add chunks to vector store
                self.vectorstore.add_documents(chunks)
            print("Successfully seeded all new documents.")
        else:
            print("No new documents to add/seed.")

    def query_stream(self, user_input: str) -> dict:
        # Retrieve source docs separately so we can display them in the CLI
        source_docs = self.retriever.invoke(user_input)
        answer_stream = self.rag_chain.stream({"input": user_input})
        return {"answer_stream": answer_stream, "context": source_docs}
