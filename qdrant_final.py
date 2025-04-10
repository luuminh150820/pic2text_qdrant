import os
import base64
import re
import io
import numpy as np
import google.generativeai as genai  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.http import models  # type: ignore
import pdf2image  # type: ignore
from PIL import Image, ImageOps  # type: ignore
from tqdm import tqdm  # Import tqdm

# Config
OUTPUT_FOLDER = "images"
QDRANT_COLLECTION = "image_text_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_TEXT_FILE = "extracted_text.txt"
VECTOR_SIZE = 768
MAX_CHUNK_SIZE = 1000
MIN_SIMILARITY_SCORE = 0.7
POPPLER_PATH = r"E:\Minh\poppler-24.08.0\Library\bin"
PDF_PATH = "input.pdf"

# Image preprocessing config
MAX_IMAGE_SIZE = (800, 800)  # Reduced max size for faster processing
BATCH_SIZE = 5  # Number of images to process in one batch

# Path for local Qdrant storage
QDRANT_LOCAL_PATH = os.path.join(os.getcwd(), "qdrant_storage")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

def preprocess_image(image_path):
    """Preprocess image for better OCR performance"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if larger than MAX_IMAGE_SIZE
        if img.width > MAX_IMAGE_SIZE[0] or img.height > MAX_IMAGE_SIZE[1]:
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        img_gray = ImageOps.grayscale(img)
        
        # Enhance contrast
        img_gray = ImageOps.autocontrast(img_gray)
        
        # Save to buffer
        buffer = io.BytesIO()
        img_gray.save(buffer, format="PNG")
        buffer.seek(0)
        
        return buffer.getvalue(), 'image/png'
    except Exception as e:
        print(f"Error preprocessing image {os.path.basename(image_path)}: {e}")
        # If preprocessing fails, return the original image
        with open(image_path, "rb") as f:
            return f.read(), f'image/{os.path.splitext(image_path)[1][1:].lower()}'

def pdf_to_images(pdf_path, output_folder):
    try:
        images = pdf2image.convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        print("PDF conversion complete")
        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

# Configure Gemini and Qdrant with local storage
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = genai.GenerativeModel("embedding-001")

# Ensure the local storage directory exists
os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)

# Initialize Qdrant with local path
#qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
qdrant_client = QdrantClient(location=":memory:")

def get_embedding(text):
    """Generate embedding for text"""
    try:
        result = embedding_model.generate_content(text)
        return result.embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def split_into_sentences(text):
    """Split text into sentences using regex pattern"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def recursive_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    """Chunk text recursively while preserving semantic boundaries"""
    if not text:
        return []

    # First try to split by paragraphs
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        elif len(paragraph) > max_chunk_size:
            # If paragraph is too large, split into sentences
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            sentences = split_into_sentences(paragraph)
            sentence_chunk = ""

            for sentence in sentences:
                if len(sentence_chunk) + len(sentence) > max_chunk_size and sentence_chunk:
                    chunks.append(sentence_chunk)
                    sentence_chunk = sentence
                else:
                    separator = " " if sentence_chunk else ""
                    sentence_chunk += separator + sentence

            if sentence_chunk:
                current_chunk = sentence_chunk
        else:
            separator = "\n\n" if current_chunk else ""
            current_chunk += separator + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def process_image_batch(image_paths):
    """Process a batch of images at once using Gemini"""
    if not image_paths:
        return {}
        
    try:
        # Prepare batch of images
        parts = []
        
        # First, add all the images with preprocessing
        for image_path in image_paths:
            image_data, mime_type = preprocess_image(image_path)
            parts.append({"inline_data": {"mime_type": mime_type, "data": image_data}})
        
        # Add the prompt text as the last part
        parts.append({"text": "Act like a text scanner. Extract text from each image in order, separated by '---IMAGE BOUNDARY---'. Extract text as it is without analyzing it and without summarizing it."})
        
        # Make a single API call for all images
        response = generation_model.generate_content(parts)
        text_results = response.text.strip()
        
        # Split the response by image boundary marker
        text_segments = text_results.split('---IMAGE BOUNDARY---')
        
        # Clean up and create results dictionary
        results = {}
        for i, (image_path, text) in enumerate(zip(image_paths, text_segments)):
            # Clean up any extra whitespace
            cleaned_text = text.strip()
            if cleaned_text:
                results[image_path] = cleaned_text
                
        return results
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fall back to processing images individually
        print("Falling back to individual processing...")
        results = {}
        for image_path in image_paths:
            try:
                text = process_single_image(image_path)
                if text:
                    results[image_path] = text
            except Exception as inner_e:
                print(f"Error processing {os.path.basename(image_path)}: {inner_e}")
        return results

def process_single_image(image_path):
    """Process a single image if batch processing fails"""
    try:
        image_data, mime_type = preprocess_image(image_path)
        
        parts = [
            {"inline_data": {"mime_type": mime_type, "data": image_data}},
            {"text": "Act like a text scanner. Extract text as it is without analyzing it and without summarizing it."}
        ]

        response = generation_model.generate_content(parts)
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text from {os.path.basename(image_path)}: {e}")
        return ""

def setup_qdrant_collection():
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        if QDRANT_COLLECTION not in [c.name for c in collections]:
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
        return False

def store_chunks_in_qdrant(chunks, metadata):
    """Store text chunks and their embeddings in Qdrant"""
    try:
        # Get embeddings for all chunks
        points = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                points.append(
                    models.PointStruct(
                        id=metadata.get("id_prefix", 0) + i,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source_image": metadata.get("source_image", "unknown"),
                            "chunk_index": i,
                            "chunk_level": metadata.get("chunk_level", "document")
                        }
                    )
                )

        # Upload points to Qdrant
        if points:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            return len(points)
        return 0
    except Exception as e:
        print(f"Error storing data in Qdrant: {e}")
        return 0

def advanced_search(query, filters=None, min_score=MIN_SIMILARITY_SCORE, limit=5):
    """Search with filtering and score thresholds"""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        # Create filter conditions
        filter_condition = None
        if filters:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
            )

        # Search with filters
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_condition
        )

        # Apply score threshold and return results
        return [
            {
                "text": r.payload["text"],
                "score": r.score,
                "source_image": r.payload["source_image"],
                "chunk_index": r.payload["chunk_index"]
            }
            for r in search_results if r.score >= min_score
        ]
    except Exception as e:
        print(f"Error in advanced search: {e}")
        return []

def retrieve_and_generate(query, filters=None):
    """Retrieve relevant context and generate an answer"""
    try:
        results = advanced_search(query, filters=filters)
        if not results:
            return "No relevant information found."

        # Combine results into context
        context = "\n\n".join([f"[From {r['source_image']}]\n{r['text']}" for r in results])
        prompt = f"Answer the following question based on the context:\nQuestion: {query}\nContext: {context}"

        response = generation_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error retrieving and generating response: {e}")
        return "Error generating response."

def main():
    # Create images folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    #image_paths = pdf_to_images(PDF_PATH, OUTPUT_FOLDER)
    print("Starting image processing...")
    # Find all images
    image_paths = [
        os.path.join(OUTPUT_FOLDER, filename)
        for filename in os.listdir(OUTPUT_FOLDER)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    if not image_paths:
        print("No images found in the 'images' folder.")
        return

    # Set up Qdrant collection
    if not setup_qdrant_collection():
        print("Failed to set up Qdrant collection.")
        return

    # Process images in batches with progress tracking
    print(f"Processing {len(image_paths)} images in batches of {BATCH_SIZE}...")
    extracted_texts = {}
    
    # Create batches
    batches = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    
    # Process each batch with progress bar
    for batch in tqdm(batches, desc="Processing image batches"):
        batch_results = process_image_batch(batch)
        extracted_texts.update(batch_results)
    
    # Show completion status
    successful_count = len(extracted_texts)
    print(f"✓ Completed text extraction: {successful_count}/{len(image_paths)} images processed successfully")

    if not extracted_texts:
        print("No text extracted from images.")
        return

    # Save combined text to file
    combined_text = "\n\n".join(extracted_texts.values())
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"✓ Extracted text saved to {OUTPUT_TEXT_FILE}")

    # Process and store text chunks
    total_chunks = 0
    chunk_id_counter = 0
    
    # Process text chunks with progress bar
    for image_path, text in tqdm(extracted_texts.items(), desc="Processing Text Chunks"):
        source_image = os.path.basename(image_path)

        # Create semantic chunks
        chunks = recursive_chunk_text(text)

        # Store chunks in Qdrant with unique IDs
        metadata = {
            "source_image": source_image,
            "chunk_level": "semantic",
            "id_prefix": chunk_id_counter
        }
        stored_chunks = store_chunks_in_qdrant(chunks, metadata)
        total_chunks += stored_chunks
        chunk_id_counter += len(chunks)

    # Show completion summary
    print(f"✓ Completed storing {total_chunks} chunks in Qdrant")
    print(f"✓ All processing tasks completed successfully!")

    # Example query
    if total_chunks > 0:
        print("\nTesting search and retrieval:")
        query = "What is the main topic of these documents?"
        print(f"Query: {query}")
        answer = retrieve_and_generate(query)
        print(f"Answer: {answer}")

        # Example filtered search
        if len(image_paths) > 1:
            source_filter = {"source_image": os.path.basename(image_paths[0])}
            print(f"\nFiltered search for image {source_filter['source_image']}:")
            filtered_answer = retrieve_and_generate(query, filters=source_filter)
            print(f"Filtered answer: {filtered_answer}")

if __name__ == "__main__":
    main()