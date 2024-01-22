import qdrant_client, os, json
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex, Document
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

embedding_llm = HuggingFaceEmbedding(
                                    model_name="BAAI/bge-small-en-v1.5",
                                    device='mps'
                                    )

def load_data(data_dir="./data/json/"):
    # reader = SimpleDirectoryReader(
    #                                 input_dir="./data/json/", 
    #                                 recursive=True
    #                                 )
    # docs = reader.load_data()
    
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(data_dir+filename, 'r') as f:
                json_data = json.load(f)
                url = json_data['url']
                title = json_data['title']
                text = json_data['text']

                document = Document(
                                    text=text,
                                    metadata={
                                        'url': url,
                                        'title': title
                                        }
                                    )
                documents.append(document)
    return documents

def chunk_docs(docs):
    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="test_store")

    pipeline = IngestionPipeline(
                                transformations=[
                                                SentenceSplitter(
                                                                chunk_size=512, 
                                                                chunk_overlap=100
                                                                ),
                                embedding_llm
                                ],
                                vector_store=vector_store
                                )
    pipeline.run(documents=docs)

    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

def build_index():
    docs = load_data()
    index = chunk_docs(docs)
    return index