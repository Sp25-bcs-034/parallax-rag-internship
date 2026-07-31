import chromadb

def test_chromadb():

    client = chromadb.Client()

    collection = client.create_collection("test_collection")

    collection.add(
        ids=["1"],
        documents=["Artificial Intelligence"],
        embeddings=[[0.1]*384],
        metadatas=[{"label":0}]
    )

    result = collection.get()

    assert len(result["ids"]) == 1