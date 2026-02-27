from pydantic import BaseModel

class StartSessionRequest(BaseModel):
    agent_type: str
    scenario: str

class StartSessionResponse(BaseModel):
    session_id: str

class MessageRequest(BaseModel):
    session_id: str
    message: str

class MessageResponse(BaseModel):
    content: str
