from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import analyze_emotion, generate_reaction
import logging
from datetime import datetime

log_filename = f"server-logs/server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

app = FastAPI()

class TextInput(BaseModel):
    text: str

# 루트 경로 설정
@app.get("/")
async def root():
    return {"message": "Server on"}

from fastapi.responses import JSONResponse

from pydantic import BaseModel
from fastapi.responses import JSONResponse

# 반환 데이터 모델 정의
class EmotionResponse(BaseModel):
    sentence: str
    predict1: str
    predict2: str
    predict3: str

class ReactionResponse(BaseModel):
    sentence: str
    response1: str
    response2: str

# 감정 분석 API
@app.post("/analyze", response_model=EmotionResponse)
async def analyze_text(input: TextInput):
    try:
        logging.info(f"Received text: {input.text}")
        result = analyze_emotion(input.text)
        return JSONResponse(content=result)
    except HTTPException as e:
        logging.error(f"HTTPException during processing: {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 반응 생성 API
@app.post("/react", response_model=ReactionResponse)
async def react_text(input: TextInput):
    try:
        logging.info(f"Received text: {input.text}")
        result = generate_reaction(input.text)
        return JSONResponse(content=result)
    except HTTPException as e:
        logging.error(f"HTTPException during processing: {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logging.info("Server shutdown.")

# uvicorn server:app --host 0.0.0.0 --port 8000