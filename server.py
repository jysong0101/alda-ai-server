from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import analyze_emotion, generate_feedback_segments, generate_reaction
import logging
from datetime import datetime
from fastapi.responses import JSONResponse
from typing import List

log_filename = f"server-logs/server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

app = FastAPI()

class TextInput(BaseModel):
    text: str

# 루트 경로 설정
@app.get("/")
async def root():
    return {"message": "Server on"}

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


class FeedbackInput(BaseModel):
    markdown_content: str

class FeedbackSegment(BaseModel):
    startIndex: int
    endIndex: int
    feedback: str

class FeedbackResponse(BaseModel):
    sentence: str
    feedback_segments: List[FeedbackSegment]

class FeedbackInput(BaseModel):
    content: str

class FeedbackSegment(BaseModel):
    startIndex: int
    endIndex: int
    feedback: str

class FeedbackResponse(BaseModel):
    feedback_segments: List[FeedbackSegment]

# 문장별 피드백 생성 API
@app.post("/generate_feedback", response_model=FeedbackResponse)
async def generate_feedback(input: FeedbackInput):
    try:
        logging.info(f"Received content for feedback: {input.content}")
        result = generate_feedback_segments(input.content)
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

