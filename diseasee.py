import os
import json
import re
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
ANYTHING_LLM_API_KEY = "KJ8NKJ9-KD84RN8-KM8E54F-HFHVJVRwiuhwihwefekjwn"

app = FastAPI(title="Disease Medical Analysis Bridge (AnythingLLM Powered)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("🚀 Disease Bridge Server Online.")
print("--- HEALTH & DISEASE INTENT MONITOR RUNNING ---")

class ChatMessage(BaseModel):
    user: str
    bot: str

class DiseaseResearchItem(BaseModel):
    source: Optional[str] = "disease_database"
    name: Optional[str] = ""
    type: Optional[str] = ""
    info: Optional[str] = ""
    symptoms: Optional[str] = ""
    medicines: Optional[str] = ""
    foods_to_avoid: Optional[str] = ""

class ChatRequest(BaseModel):
    user_prompt: str
    history: Optional[List[ChatMessage]] = []
    research_data: Optional[List[DiseaseResearchItem]] = []


def build_disease_context(research_data: List[DiseaseResearchItem]) -> str:
    if not research_data:
        return ""
    blocks = []
    for d in research_data:
        lines = []
        
        # Only include structural labels if content actually exists
        if d.name and d.name.strip():
            lines.append(f"DISEASE IDENTITY: {d.name.strip()}")
        if d.type and d.type.strip():
            lines.append(f"MEDICAL CATEGORY: {d.type.strip()}")
        if d.info and d.info.strip():
            lines.append(f"OVERVIEW INFO: {d.info.strip()}")
        if d.symptoms and d.symptoms.strip():
            lines.append(f"IDENTIFIED SYMPTOMS: {d.symptoms.strip()}")
        if d.medicines and d.medicines.strip():
            lines.append(f"RECOMMENDED MEDICINES/CARE: {d.medicines.strip()}")
        if d.foods_to_avoid and d.foods_to_avoid.strip():
            lines.append(f"CRITICAL FOODS TO AVOID: {d.foods_to_avoid.strip()}")
            
        if lines:
            blocks.append("\n".join(lines))
            
    return "\n\n".join(blocks)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):

    async def event_generator():
        # Build clean descriptive medical context profile from incoming Django lists
        context_text = build_disease_context(req.research_data)

        system_instruction = (
            "You are an expert AI medical assistant. Using the provided DISEASE DATA CONTEXT, "
            "directly answer the USER QUESTION. Break down the symptoms, explain what the disease is, "
            "highlight what medicines or clinical care are relevant, and clearly detail what foods "
            "they MUST avoid to protect their recovery process. Speak clearly, supportive, and informative. if no context is provided proeprly , say 'I'm sorry, but I don't have the necessary information to answer that question.'"
        )
        
        full_message = f"{system_instruction}\n\n"
        if context_text:
            full_message += f"DISEASE DATA CONTEXT FROM MEDICAL DATABASE:\n{context_text}\n\n"
        
        full_message += f"USER HEALTH QUESTION: {req.user_prompt}"

        payload = {
            "message": full_message,
            "mode": "chat",
            "stream": True 
        }
        
        headers = {
            "Authorization": f"Bearer {ANYTHING_LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        # Use async httpx client to manage long running text streams
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
                                
                                # Process target variant mapping protocols
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
                print(f"❌ Stream breakdown on backend runtime: {traceback.format_exc()}")
                yield f"data: {json.dumps({'token': f'Medical processing error encountered: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # Assigned to unique Port 8003 to prevent port clashing with foodie service
    uvicorn.run(app, host="0.0.0.0", port=8003)