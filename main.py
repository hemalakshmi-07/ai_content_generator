from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variables
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face model endpoint
HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

@app.get("/")
def read_root():
    return {"message": "Backend running!"}

@app.post("/generate")
async def generate(request: dict):
    try:
        # Build the prompt from user inputs
        content_type = request.get('type', 'blog')
        user_prompt = request.get('prompt', '')
        tone = request.get('tone', 'professional')
        audience = request.get('audience', 'general')
        language = request.get('language', 'english')
        word_count = request.get('word_count', 500)
        
        # Construct detailed prompt for better results
        full_prompt = f"""Write a {content_type} about: {user_prompt}

Requirements:
- Tone: {tone}
- Target Audience: {audience}
- Language: {language}
- Approximate length: {word_count} words
- Make it engaging and high quality

Content:"""
        
        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload for Hugging Face API
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_length": int(word_count) * 2,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        # Make request to Hugging Face API
        response = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract generated text from response
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Remove the prompt from the response to get just the generated content
                if full_prompt in generated_text:
                    content = generated_text.split(full_prompt)[-1].strip()
                else:
                    content = generated_text
            else:
                content = "Error: Unexpected response format"
            
            return {
                "content": content,
                "success": True
            }
        else:
            error_msg = response.text if response.text else f"HTTP {response.status_code}"
            return {
                "error": f"API Error: {error_msg}",
                "success": False
            }
    
    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout. Please try again.",
            "success": False
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Connection error: {str(e)}",
            "success": False
        }
    except Exception as e:
        return {
            "error": f"Error: {str(e)}",
            "success": False
        }
