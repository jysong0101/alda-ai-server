from fastapi import FastAPI, HTTPException
from llama_cpp import Dict
from pydantic import BaseModel
from model import analyze_emotion, generate_feedback_segments, generate_reaction
from rag_model import generate_feedback_with_rag_chain, add_diary_entry_to_db, load_rag_files

import logging
from datetime import datetime
from fastapi.responses import JSONResponse
from typing import List

log_filename = f"server-logs/server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

app = FastAPI()


class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Server on"}

class EmotionResponse(BaseModel):
    sentence: str
    predict1: str
    predict2: str
    predict3: str

class ReactionResponse(BaseModel):
    sentence: str
    response1: str
    response2: str

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
    text: str

class FeedbackSegment(BaseModel):
    startIndex: int
    endIndex: int
    feedback: str

class FeedbackResponse(BaseModel):
    feedback_segments: List[FeedbackSegment]

@app.post("/generate_feedback", response_model=FeedbackResponse)
async def generate_feedback(input: FeedbackInput):
    try:
        logging.info(f"Received content for feedback: {input.text}")
        result = generate_feedback_segments(input.text)
        return JSONResponse(content=result) 
    except HTTPException as e:
        logging.error(f"HTTPException during processing: {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 다이어리 추가 입력 데이터 구조
class DiaryEntry(BaseModel):
    memberId: str
    diaryId: str
    diaryTitle: str
    diaryContents: str
    diaryEntryDate: str  # ISO 형식 날짜
    diaryFeedbacks: List[Dict] = []

# API: 다이어리 추가
@app.post("/add_diary_entry")
async def add_diary_entry(entry: DiaryEntry):
    try:
        logging.info(f"Received diary entry for memberId {entry.memberId}")
        add_diary_entry_to_db(entry.memberId, entry.dict())
        return {"message": f"Diary entry for memberId {entry.memberId} added successfully."}
    except Exception as e:
        logging.error(f"Error adding diary entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to add diary entry.")

# 피드백 입력 및 출력 데이터 구조
class FeedbackInput(BaseModel):
    text: str

class FeedbackSegment(BaseModel):
    startIndex: int
    endIndex: int
    feedback: str

class FeedbackResponse(BaseModel):
    feedback_segments: List[FeedbackSegment]

# API: RAG 기반 피드백 생성
@app.post("/generate_feedback_rag", response_model=FeedbackResponse)
async def generate_feedback_rag(input: FeedbackInput, member_id: str):
    try:
        logging.info(f"Generating feedback with RAG for memberId {member_id} and content: {input.text}")
        result = generate_feedback_with_rag_chain(member_id, input.text)
        return JSONResponse(content=result)
    except HTTPException as e:
        logging.error(f"HTTPException during processing: {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Feedback generation with RAG failed.")

if __name__ == "__main__":
    import uvicorn

    logging.info("Initializing server and loading RAG files.")
    load_rag_files()  # 서버 시작 시 RAG 파일 로드
    logging.info("Starting server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)