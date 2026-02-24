from fastapi import FastAPI, HTTPException
from .session_manager import SessionManager
from .schemas import StartSessionRequest, StartSessionResponse, MessageRequest, MessageResponse
from pathlib import Path
import json

app = FastAPI()
sessions = SessionManager()

@app.post("/start_session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    session_id = sessions.start_session(req.agent_type, req.scenario)
    return StartSessionResponse(session_id=session_id)

@app.post("/message", response_model=MessageResponse)
def message(req: MessageRequest):
    session = sessions.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    print("USER MESSAGE:", req.message)
    # Add user message to context
    sessions.add_user_message(session, req.message)
    # Get agent response
    response = sessions.agent_respond(session, session_id=req.session_id)
    if response is None:
        response = "Agent did not respond."
    return MessageResponse(content=response)

@app.get("/trajectory/{session_id}")
def get_trajectory(session_id: str):
    conversation_path = Path("data") / session_id / "conversation.json"
    if not conversation_path.exists():
        # raise HTTPException(status_code=404, detail="Trajectory not found")
        return [{"role": "system", "content": "Error 404: No conversation found."}]
    with open(conversation_path, "r", encoding="utf-8") as f:
        conversation = json.load(f)
        
    print(conversation)
    return conversation
