from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_ai_response(message: str, history: list = []) -> str:
    # Build message list with history
    messages = [
        {"role": "system", "content": "You are SmartHub AI, a helpful assistant."}
    ]
    
    # Add conversation history
    for msg in history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Add current message
    messages.append({
        "role": "user",
        "content": message          
    })
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    # Extract token usage from response
    usage = response.usage

    return {
        "reply": response.choices[0].message.content,
        "token_usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
    }

def summarize_text(text: str) -> str:

    prompt = f"""
    Summarize the following document clearly.

    Document:
    {text[:8000]}

    Summary:
    """
    result = get_ai_response(prompt)
    return result["reply"]