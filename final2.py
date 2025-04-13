import os
import base64
import google.generativeai as genai
from qdrant_client import QdrantClient, models
import traceback
import uuid

#WORKS BUT WRONG LENGTH

#os.environ["HTTP_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"
#os.environ["HTTPS_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"

# Config
OUTPUT_FOLDER = "images"
QDRANT_COLLECTION = "pdf_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_TEXT_FILE = "extracted_text.txt"
VECTOR_SIZE = 768

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def get_embedding(text):
    """Generate embedding for text using embeddings API"""
    try:
        embedding_result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        if hasattr(embedding_result, "embedding"):
            if callable(embedding_result.embedding):
                embedding_vector = list(embedding_result.embedding())
            else:
                embedding_vector = list(embedding_result.embedding)
            return embedding_vector
        if hasattr(embedding_result, "values"):
            if callable(embedding_result.values):
                embedding_vector = list(embedding_result.values())
            else:
                embedding_vector = list(embedding_result.values)
            return embedding_vector
        try:
            return list(embedding_result)
        except:
            pass
        print("Unable to extract embedding vector from result")
        return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_image_to_text(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        text_prompt = "Act like a text scanner. Extract text as it is without analyzing it and without summarizing it."
        parts = [
            {"inline_data": {"mime_type": "image/png", "data": image_data}},
            {"text": text_prompt}
        ]
        response = model.generate_content(parts)
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""

def setup_qdrant_collection(qdrant_client):
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # If collection exists, recreate it to avoid schema conflicts
        if QDRANT_COLLECTION in collection_names:
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
            print(f"Deleted existing collection: {QDRANT_COLLECTION}")
        
        # Create collection with a simple non-named vector configuration
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {QDRANT_COLLECTION}")
        return True
    except Exception as e:
        print(f"Error setting up Qdrant collection: {e}")
        traceback.print_exc()
        return False

def store_in_qdrant(qdrant_client, texts, collection_name, ids):
    try:
        points = []
        for i, text in enumerate(texts):
            embedding = get_embedding(text)
            if embedding:
                print(f"Embedding type: {type(embedding)}, Length: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
                # Use vector parameter for older Qdrant client versions
                points.append(
                    models.PointStruct(
                        id=ids[i],
                        vector=embedding,
                        payload={"text": text}
                    )
                )
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Successfully stored {len(points)} points in Qdrant")
        return True
    except Exception as e:
        print(f"Error storing data in Qdrant: {e}")
        traceback.print_exc()
        return False

def retrieve_similar_points(qdrant_client, query_text, collection_name, limit=3):
    """Alternative method to retrieve similar points without using search"""
    try:
        # Get all points from the collection
        all_points = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,  # Adjust based on your expected collection size
            with_payload=True,
            with_vectors=True
        )[0]  # scroll returns (points, offset)
        
        if not all_points:
            print("No points found in collection")
            return []
        
        # Generate query embedding
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            print("Failed to generate query embedding")
            return []
        
        # Calculate cosine similarity manually
        def cosine_similarity(vec1, vec2):
            # Simple cosine similarity implementation
            if len(vec1) != len(vec2):
                print(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0
                
            return dot_product / (norm_a * norm_b)
        
        # Calculate similarities and sort
        similarities = []
        for point in all_points:
            if hasattr(point, 'vector') and point.vector:
                sim = cosine_similarity(query_embedding, point.vector)
                similarities.append((point, sim))
            else:
                print(f"Point {point.id} has no vector attribute")
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return [item[0] for item in similarities[:limit]]
    except Exception as e:
        print(f"Error in retrieve_similar_points: {e}")
        traceback.print_exc()
        return []

def retrieve_and_generate(qdrant_client, query, collection_name):
    try:
        # Use our custom retrieval method instead of search
        similar_points = retrieve_similar_points(qdrant_client, query, collection_name)
        
        if not similar_points:
            return "No relevant information found."

        context = " ".join([p.payload["text"] for p in similar_points])
        prompt = f"Answer the following question based on the context: {query}\nContext: {context}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error retrieving and generating response: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    image_paths = [os.path.join(OUTPUT_FOLDER, filename) for filename in os.listdir(OUTPUT_FOLDER) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'))]
    if not image_paths:
        print("No image files found in the 'images' folder.")
        exit(1)
    extracted_text = ""
    for image_path in image_paths:
        extracted_text += process_image_to_text(image_path) + " "
    try:
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)
        print(f"Extracted text saved to {OUTPUT_TEXT_FILE}")
    except Exception as e:
        print(f"Error saving extracted text to file: {e}")
    text_chunks = [extracted_text[i:i + 1000] for i in range(0, len(extracted_text), 1000)]
    ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
    qdrant_client = QdrantClient(location=":memory:")
    if setup_qdrant_collection(qdrant_client) and store_in_qdrant(qdrant_client, text_chunks, QDRANT_COLLECTION, ids):
        query = "What is the main topic?"
        answer = retrieve_and_generate(qdrant_client, query, QDRANT_COLLECTION)
        if answer:
            print(f"Answer: {answer}")
        
        # For final check - use our custom retrieval method
        try:
            similar_to_first = retrieve_similar_points(qdrant_client, text_chunks[0], QDRANT_COLLECTION, limit=1)
            if similar_to_first:
                print(f"Sample retrieved text: {similar_to_first[0].payload['text'][:100]}...")
        except Exception as e:
            print(f"Error in final check: {e}")
