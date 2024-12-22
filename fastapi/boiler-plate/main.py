from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def get_root_handler():
    return {"message": "Your made GET on /"}

@app.get("/health")
def get_health_handler():
    return {"message": "I'm still alive"}

# curl -X POST localhost:8000/?data=asdf
@app.post("/")
def post_root_handler(data):
    return {"message": data}

