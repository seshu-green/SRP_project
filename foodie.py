import json
import httpx
import asyncio
import traceback
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ---------------- CONFIG ----------------
#replace this with your AnythingLLM endpoint and API key
ANYTHING_LLM_URL = "http://localhost:3001/api/v1/workspace/foodie/chat"
ANYTHING_LLM_API_KEY = "KJ8NKJ9-KD84RN8-KM8E54F-HFHVJVRfmbekjnf"

app = FastAPI(title="Foodbot System (AnythingLLM Powered)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("🚀 Bridge is ready. Connecting via AnythingLLM...")
print("--- FOODBOT ONLINE (ANYTHINGLLM BACKEND) ---")

class ChatMessage(BaseModel):
    user: str
    bot: str

class ResearchItem(BaseModel):
    # Added info and type to match the Django payload
    type: Optional[str] = "food"
    info: Optional[str] = ""
    item: Optional[str] = ""
    variant: Optional[str] = ""
    method: Optional[str] = ""
    nutrients: Optional[str] = ""
    benefits: Optional[str] = ""
    hazards: Optional[str] = ""

class ChatRequest(BaseModel):
    user_prompt: str
    history: Optional[List[ChatMessage]] = []
    research_data: Optional[List[ResearchItem]] = []

def build_context(research_data: List[ResearchItem]) -> str:
    if not research_data:
        return ""
    blocks = []
    for r in research_data:
        # Constructing a descriptive block including Type and Info
        block = (
            f"Type: {r.type}\n"
            f"Info: {r.info}\n"
            f"Food Item: {r.item}\n"
            f"Variant/type: {r.variant}\n"
            f"Method: {r.method}\n"
            f"Nutrients: {r.nutrients}\n"
            f"Benefits: {r.benefits}\n"
            f"Hazards: {r.hazards}"
        )
        blocks.append(block.strip())
    return "\n\n".join(blocks)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):

    async def event_generator():
        context_text = build_context(req.research_data)

        system_instruction = (
            "u are a food assistant.Using DATA CONTEXT please answer according to user asked QUESTION directly or closely. Try to explain based on the question. DATA CONTEXT:"
        )
        
        full_message = f"{system_instruction}"
        if context_text:
            full_message += f"DATA CONTEXT:\n{context_text}\n\n"
        
        full_message += f"USER QUESTION: {req.user_prompt}"

        payload = {
            "message": full_message,
            "mode": "chat",
            "stream": True 
        }
        
        headers = {
            "Authorization": f"Bearer {ANYTHING_LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        # Use async httpx client to bypass blocking issues
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", ANYTHING_LLM_URL, json=payload, headers=headers) as response:
                    async for line in response.aiter_lines():
                        if line:
                            decoded_line = line.strip()
                            
                            if not decoded_line or decoded_line == ":" or decoded_line.startswith(":ping"):
                                continue
                            
                            if decoded_line.startswith("data:"):
                                decoded_line = decoded_line[5:].strip()
                            
                            try:
                                json_data = json.loads(decoded_line)
                                token = None
                                
                                # Priority extraction for streaming tokens
                                if "textResponse" in json_data:
                                    token = json_data.get("textResponse")
                                elif "content" in json_data:
                                    token = json_data.get("content")
                                elif "choices" in json_data and len(json_data["choices"]) > 0:
                                    choice = json_data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        token = choice["delta"]["content"]
                                    elif "text" in choice:
                                        token = choice["text"]
                                
                                if token:
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                            
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"❌ Error during stream processing: {traceback.format_exc()}")
                yield f"data: {json.dumps({'token': f'Error processing text: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)