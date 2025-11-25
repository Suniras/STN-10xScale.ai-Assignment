from src.path_setup import *

import google.generativeai as genai
from src.config import GOOGLE_API_KEY, GEMINI_MODEL

def get_gemini_llm():
    """Configure and return Gemini model"""
    genai.configure(api_key=GOOGLE_API_KEY)
    
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            "temperature": 0.1,  # Low temperature for factual responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
    )
    return model

def generate_answer(context: str, query: str) -> str:
    model = get_gemini_llm()
    
    prompt = f"""You are an expert HR assistant for NebulaGears company. Your job is to answer employee questions based ONLY on the provided policy documents.


**Context (Policy Documents):**
{context}

**User Question:**
{query}

**Your Answer:**
(Provide a clear, specific answer that prioritizes the most relevant policy for the user's role. Always cite the source document.)
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"