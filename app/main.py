import os
os.environ.pop("SSL_CERT_FILE", None)

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agent.graph import app as travel_graph
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# ✅ Allow CORS (only needed if frontend runs on separate domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static files from /frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class TravelQuery(BaseModel):
    message: str

# ✅ POST endpoint to process travel query
@app.post("/plan-trip")
def plan_trip(query: TravelQuery):
    try:
        stream = travel_graph.stream(
            {"messages": [HumanMessage(content=query.message)]},
            config={"configurable": {"thread_id": "user-thread"}},
            stream_mode="values"
        )
        result = []
        for event in stream:
            content = event["messages"][-1].content
            if content:
                result.append(content)

        return {"response": result[-1] if result else "No response from model."}

    except Exception as e:
        print("❌ Error in /plan-trip:", str(e))
        return {"response": "⚠️ Sorry, something went wrong on the server."}

# ✅ Serve the UI
@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")
