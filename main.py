import os
import base64
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import tempfile
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Model will be downloaded on first run
MODEL_NAME = "OpenFace"
MODEL_BACKEND = "opencv"


def download_image_from_base64(base64_string):
    """Convert base64 string to image"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"Error decoding image: {str(e)}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL_NAME}), 200

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
 
    try:
        img = None
        temp_path = None
        
        # --- NEW CODE: Handle Multipart/Form-Data (Java Client's method) ---
        if 'file' in request.files:
            # Image sent as a file upload (multipart/form-data)
            file = request.files['file']
            
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name # Use temp_path directly for DeepFace
                
            logging.info("Received image via multipart/form-data.")
            # DeepFace will read from temp_path later. Skip further image loading.
            
        # --- Existing Code: Handle JSON (for other clients) ---
        elif request.is_json:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Get image from either base64 or GCS
            if 'image' in data:
                img = download_image_from_base64(data['image'])
            else:
                return jsonify({"error": " 'image'  required in JSON"}), 400
            
            if img is None:
                return jsonify({"error": "Failed to process image from JSON"}), 400
            
            # Save to temporary file for DeepFace
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                cv2.imwrite(temp_file.name, img)
                temp_path = temp_file.name # Use temp_path for DeepFace
        
        # --- UNCHANGED: DeepFace Processing Logic ---
        else:
            return jsonify({"error": "Unsupported Content-Type. Expected JSON or multipart/form-data"}), 400
        
        if temp_path is None:
            return jsonify({"error": "Failed to prepare image for embedding"}), 400

        try:
            # Generate embedding using DeepFace
            logging.info(f"Generating embedding with {MODEL_NAME} from {temp_path}")
            
            embedding_objs = DeepFace.represent(
                img_path=temp_path, # Use the path where the image was saved
                model_name=MODEL_NAME,
                detector_backend=MODEL_BACKEND,
                enforce_detection=True,
                align=True
            )
            
            # ... (Rest of the DeepFace logic is the same) ...
            if not embedding_objs or len(embedding_objs) == 0:
                os.unlink(temp_path)
                return jsonify({"error": "No face detected in image"}), 400
            
            embedding = embedding_objs[0]["embedding"]
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return jsonify({
                "embedding": embedding,
                "dimension": len(embedding),
                "model": MODEL_NAME
            }), 200
            
        except ValueError as ve:
            os.unlink(temp_path)
            logging.error(f"Face detection error: {str(ve)}")
            return jsonify({"error": "No face detected in image"}), 400
        except Exception as e:
            os.unlink(temp_path)
            logging.error(f"Embedding generation error: {str(e)}")
            return jsonify({"error": f"Failed to generate embedding: {str(e)}"}), 500
            
    except Exception as e:
        logging.error(f"Request processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=8080, debug=True)