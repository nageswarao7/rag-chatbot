import chromadb

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chromadb")
        self.collection = self.client.get_or_create_collection(name="rag_documents")

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """Add documents to the vector store."""
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5):
        """Query the vector store for similar documents."""
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def count(self):
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear_all(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(name="rag_documents")
        self.collection = self.client.get_or_create_collection(name="rag_documents")
        return True

# Singleton instance
vector_store = VectorStore()
