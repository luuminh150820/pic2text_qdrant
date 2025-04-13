import os
import base64
import google.generativeai as genai
from qdrant_client import QdrantClient, models
import traceback
import uuid
import time

# Configuration
OUTPUT_FOLDER = "images"
QDRANT_COLLECTION = "pdf_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_TEXT_FILE = "extracted_text.txt"
VECTOR_SIZE = 768

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = genai.get_model("models/embedding-001")

def get_embedding(text):
    """Generate embedding for text using Google's embedding API"""
    if not text or len(text.strip()) == 0:
        print("Warning: Empty text passed to get_embedding")
        return [0.0] * VECTOR_SIZE
    
    try:
        # Use the embedding model to get embeddings
        result = embedding_model.embed_content(
            content=text,
            task_type="retrieval_document"
        )
        
        # Extract embedding values based on the structure of the response
        embedding_values = None
        
        # Debug information
        print(f"Embedding result type: {type(result)}")
        print(f"Embedding result dir: {dir(result)}")
        
        # Try different ways to access the embedding values
        if hasattr(result, "embeddings") and result.embeddings:
            embedding_values = result.embeddings[0]
            print("Accessed via .embeddings[0]")
        elif hasattr(result, "embedding") and result.embedding:
            embedding_values = result.embedding
            print("Accessed via .embedding")
        elif hasattr(result, "values") and result.values:
            embedding_values = result.values
            print("Accessed via .values")
        
        # Handle case where result may be directly the values
        if embedding_values is None and isinstance(result, (list, tuple)):
            embedding_values = result
            print("Result is directly a list/tuple")
        
        # Convert to list if necessary
        if embedding_values is not None:
            if callable(getattr(embedding_values, "__iter__", None)):
                embedding_values = list(embedding_values)
            
            # Verify length
            if len(embedding_values) == VECTOR_SIZE:
                print(f"Successfully generated embedding with length: {len(embedding_values)}")
                return embedding_values
            else:
                print(f"Warning: Embedding length mismatch. Got {len(embedding_values)}, expected {VECTOR_SIZE}")
                
                # If we have a nested structure (sometimes the API returns this)
                if len(embedding_values) == 1 and isinstance(embedding_values[0], (list, tuple)) and len(embedding_values[0]) == VECTOR_SIZE:
                    print("Found correct embedding in nested structure")
                    return list(embedding_values[0])
        
        # Last resort - try direct json access
        if hasattr(result, "_json") and result._json:
            try:
                if "embedding" in result._json:
                    vals = result._json["embedding"]
                    if len(vals) == VECTOR_SIZE:
                        print("Extracted from _json['embedding']")
                        return vals
                elif "values" in result._json:
                    vals = result._json["values"]
                    if len(vals) == VECTOR_SIZE:
                        print("Extracted from _json['values']")
                        return vals
            except (KeyError, AttributeError) as e:
                print(f"Error accessing _json: {e}")
        
        # If we reached here, we couldn't extract a valid embedding
        print("Failed to extract valid embedding from result")
        print(f"Fallback: Creating random embedding for debugging")
        
        # For testing purposes, generate a random vector of the correct size
        import random
        return [random.random() for _ in range(VECTOR_SIZE)]
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        
        # For testing purposes, generate a zero vector of correct size
        return [0.0] * VECTOR_SIZE

def process_image_to_text(image_path):
    """Extract text from an image using Gemini model"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        text_prompt = "Act like a text scanner. Extract text as it is without analyzing it and without summarizing it."
        parts = [
            {"mime_type": "image/jpeg", "data": image_data},
            {"text": text_prompt}
        ]
        
        response = model.generate_content(parts)
        extracted_text = response.text.strip()
        print(f"Extracted {len(extracted_text)} characters from {image_path}")
        return extracted_text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        traceback.print_exc()
        return ""

def setup_qdrant_collection(qdrant_client):
    """Create or reset Qdrant collection"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Delete existing collection if it exists
        if QDRANT_COLLECTION in collection_names:
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
            print(f"Deleted existing collection: {QDRANT_COLLECTION}")
        
        # Create collection with correct vector configuration
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
    """Store text chunks and their embeddings in Qdrant"""
    try:
        # Process in smaller batches to avoid memory issues
        batch_size = 5
        total_texts = len(texts)
        successful_points = 0
        
        for batch_start in range(0, total_texts, batch_size):
            batch_end = min(batch_start + batch_size, total_texts)
            batch_texts = texts[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            
            points = []
            for i, text in enumerate(batch_texts):
                if not text.strip():
                    print(f"Skipping empty text at index {batch_start + i}")
                    continue
                    
                # Get embedding for text
                embedding = get_embedding(text)
                
                # Verify embedding is correct length
                if embedding and len(embedding) == VECTOR_SIZE:
                    point_id = batch_ids[i]
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={"text": text, "id": str(point_id)}
                        )
                    )
                else:
                    print(f"Skipping text at index {batch_start + i} due to invalid embedding")
            
            if points:
                # Upload batch to Qdrant
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                successful_points += len(points)
                print(f"Stored batch of {len(points)} points in Qdrant (total: {successful_points}/{total_texts})")
                
                # Small delay to avoid API rate limits if needed
                time.sleep(0.5)
        
        return successful_points > 0
    except Exception as e:
        print(f"Error storing data in Qdrant: {e}")
        traceback.print_exc()
        return False

def search_similar_points(qdrant_client, query_text, collection_name, limit=3):
    """Search for similar text chunks using Qdrant search"""
    try:
        # Get embedding for query
        query_embedding = get_embedding(query_text)
        
        if not query_embedding or len(query_embedding) != VECTOR_SIZE:
            print(f"Invalid query embedding, length: {len(query_embedding) if query_embedding else 'None'}")
            return []
        
        # Search using vector similarity
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        print(f"Found {len(search_results)} similar documents")
        return search_results
    except Exception as e:
        print(f"Error searching in Qdrant: {e}")
        traceback.print_exc()
        return []

def retrieve_and_generate(qdrant_client, query, collection_name):
    """Retrieve relevant information and generate a response"""
    try:
        # Search for similar documents
        similar_points = search_similar_points(qdrant_client, query, collection_name)
        
        if not similar_points:
            return "No relevant information found."

        # Build context from similar documents
        context_parts = []
        for p in similar_points:
            text = p.payload.get("text", "")
            score = p.score if hasattr(p, "score") else "unknown"
            context_parts.append(f"[Document {p.id} (similarity: {score})]: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using context
        prompt = f"""
        Based on the following context, answer this question: {query}
        
        Context:
        {context}
        
        Please provide a concise, accurate answer based only on the information in the context.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error retrieving and generating response: {e}")
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def validate_collection(qdrant_client, collection_name, text_sample):
    """Validate the collection by searching for a sample text"""
    try:
        print("\n--- Validating Collection ---")
        embedding = get_embedding(text_sample)
        if not embedding or len(embedding) != VECTOR_SIZE:
            print(f"Validation failed: Invalid embedding for sample text")
            return False
            
        print(f"Sample embedding length: {len(embedding)}")
        
        # Try to retrieve the sample point
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=1
        )
        
        if results:
            print(f"Validation successful: Found similar document with ID {results[0].id}")
            text_preview = results[0].payload.get("text", "")[:100]
            print(f"Sample retrieved text: {text_preview}...")
            return True
        else:
            print("Validation failed: No results found")
            return False
    except Exception as e:
        print(f"Validation error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check for images
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")
    
    # Get all image files
    image_paths = [
        os.path.join(OUTPUT_FOLDER, filename) 
        for filename in os.listdir(OUTPUT_FOLDER) 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'))
    ]
    
    if not image_paths:
        print(f"No image files found in the '{OUTPUT_FOLDER}' folder.")
        exit(1)
    
    # Process all images and extract text
    print(f"Processing {len(image_paths)} images...")
    all_extracted_text = ""
    
    for image_path in image_paths:
        extracted_text = process_image_to_text(image_path)
        all_extracted_text += extracted_text + " "
    
    # Save extracted text to file
    try:
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as text_file:
            text_file.write(all_extracted_text)
        print(f"Extracted text saved to {OUTPUT_TEXT_FILE}")
    except Exception as e:
        print(f"Error saving extracted text to file: {e}")
        traceback.print_exc()
    
    # Split text into chunks
    text_chunks = [all_extracted_text[i:i + 1000] for i in range(0, len(all_extracted_text), 1000)]
    ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
    
    print(f"Split text into {len(text_chunks)} chunks")
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(location=":memory:")
    
    # Setup collection and store data
    if setup_qdrant_collection(qdrant_client) and store_in_qdrant(qdrant_client, text_chunks, QDRANT_COLLECTION, ids):
        # Validate collection
        if text_chunks:
            validate_collection(qdrant_client, QDRANT_COLLECTION, text_chunks[0])
        
        # Run a sample query
        print("\n--- Sample Query Results ---")
        query = "What is the main topic?"
        print(f"Query: {query}")
        answer = retrieve_and_generate(qdrant_client, query, QDRANT_COLLECTION)
        if answer:
            print(f"Answer: {answer}")
        
        # Interactive query mode (optional)
        interactive = False
        if interactive:
            print("\n--- Interactive Query Mode ---")
            print("Enter a question to query the document (or 'exit' to quit):")
            
            while True:
                user_query = input("> ")
                if user_query.lower() in ('exit', 'quit', 'q'):
                    break
                    
                if user_query.strip():
                    answer = retrieve_and_generate(qdrant_client, user_query, QDRANT_COLLECTION)
                    print(f"Answer: {answer}\n")
    else:
        print("Failed to set up collection or store data. Exiting.")