from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def read_root():
    return {"message": "Backend running!"}

@app.post("/generate")
async def generate(request: dict):
    try:
        prompt = f"Write {request.get('type', 'blog')} content about: {request.get('prompt', '')}. Tone: {request.get('tone', 'professional')}. For audience: {request.get('audience', 'general')}. Language: {request.get('language', 'english')}. Target length: {request.get('word_count', 500)} words. Make it high quality and engaging."
        response = model.generate_content(prompt)
        return {"content": response.text}
    except Exception as e:
        return {"error": str(e)}
