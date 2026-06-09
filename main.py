from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

@app.get("/")
def read_root():
    return {"message": "Backend running!"}

@app.post("/generate")
async def generate(request: dict):
    try:
        content_type = request.get('type', 'blog')
        user_prompt = request.get('prompt', '')
        tone = request.get('tone', 'professional')
        audience = request.get('audience', 'general')
        language = request.get('language', 'english')
        word_count = request.get('word_count', 500)
        
        if not user_prompt:
            return {"error": "Prompt is required", "success": False}
        
        full_prompt = f"""Write a {content_type} about: {user_prompt}

Requirements:
- Tone: {tone}
- Target Audience: {audience}
- Language: {language}
- Approximate length: {word_count} words
- Make it high quality and engaging"""
        
        response = model.generate_content(
            full_prompt,
            generation_config={
                'max_output_tokens': int(word_count) * 2,
                'temperature': 0.7,
                'top_p': 0.9,
            }
        )
        
        if response.text:
            return {
                "content": response.text,
                "success": True,
                "type": content_type
            }
        else:
            return {"error": "Failed to generate content", "success": False}
    
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return {"error": "API quota exceeded. Try again later.", "success": False}
        else:
            return {"error": f"Error: {error_msg}", "success": False}
