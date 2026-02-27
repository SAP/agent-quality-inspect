# Tool Sandbox Agent

**Paper:** [Tool Sandbox (arXiv)](https://arxiv.org/abs/2408.04682)
**Original Repo:** [Tool Sandbox GitHub](https://github.com/apple/ToolSandbox/tree/main)

---

## Installation

1. **Install**
   ```bash
   cd <parent_dir>/agent_runners/ToolSandbox
   python3.10 -m venv .toolsandboxenv
   source .toolsandboxenv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
    - Rename `.env copy` to `.env` inside `ToolSandbox/tool_sandbox/roles/.env copy`.
    - For Azure make sure the endpoint is from Azure Studio, i.e. it doesn't end with /models.
    - For RAPID_API_KEY, create a new account and go to url to get your API_KEY: https://rapidapi.com/developer/dashboard
    - Subscribe to the following endpoints inside RAPID's website:
      - https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-finance-data
      - https://rapidapi.com/alexanderxbx/api/maps-data
      - https://rapidapi.com/trueway/api/trueway-geocoding 
      - https://rapidapi.com/weatherapi/api/weatherapi-com
      - https://rapidapi.com/not-null-solutions1-not-null-solutions-default/api/currency-converter18
  
3. **Start the API Server**
    ```bash
    uvicorn fastapi_server.main:app --reload --port 8000
    ```

---

## Grading Notes Dataset

Grading notes is located in `ToolSandbox/fastapi_server/filtered_tool_sandbox_gNotes_final.json`. It only has positive grading notes.

---

## API Endpoints

| Endpoint                                      | Method | Description                              |
|------------------------------------------------|--------|------------------------------------------|
| `/start_session`                              | POST   | Start a new conversation session         |
| `/message`                                    | POST   | Send a message in an existing session    |
| `/trajectory/{session_id}`                    | GET    | Get the full conversation trajectory     |

---

### Start Session
**Endpoint:** `/start_session`  
**Method:** `POST`  
**Description:** Starts a new conversation session with a specified agent type and scenario.

**Request Body:**
```json
{
    "agent_type": "string",
    "scenario": "string"
}
```

**Response:**
```json
{
    "session_id": "unique_session_id"
}
```

---

### Send Message
**Endpoint:** `/message`  
**Method:** `POST`  
**Description:** Sends a message to an existing conversation session and receives the agent's response.

**Request Body:**
```json
{
    "session_id": "string",
    "message": "Your message here"
}
```

**Response:**
```json
{
    "content": "Agent's response here"
}
```

---

### Get Trajectory
**Endpoint:** `/trajectory/{session_id}`  
**Method:** `GET`  
**Description:** Retrieves the full conversation trajectory for a given session ID. The trajectory is stored as a JSON file in the data folder.

**Path Parameter:**
- `session_id`: The unique session identifier

**Response:**
```json
{
    "conversation": [
        {
            "role": "user",
            "content": "User message"
        },
        {
            "role": "agent", 
            "content": "Agent response"
        }
    ]
}
```
