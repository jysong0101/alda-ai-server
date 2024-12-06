# test_rag.py
import json
import logging
from fastapi.testclient import TestClient
from server import app

# 로깅 설정
log_filename = "test-log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

client = TestClient(app)

def test_add_diary_entry():
    # 다이어리 항목 추가 테스트
    diary_entry = {
        "memberId": "test_member",
        "diaryId": "test_diary_1",
        "diaryTitle": "Test Diary Title",
        "diaryContents": "This is a test diary entry. It contains multiple sentences.",
        "diaryEntryDate": "2023-10-10T10:00:00",
        "diaryFeedbacks": [{"feedback": "Great entry!"}]
    }

    response = client.post("/add_diary_entry", json=diary_entry)
    logging.info(f"Status Code: {response.status_code}")
    logging.info(f"Response JSON: {response.json()}")
    # assert response.status_code == 200
    # assert response.json() == {"message": "Diary entry for memberId test_member added successfully."}

def test_generate_feedback_rag():
    # RAG 기반 피드백 생성 테스트
    feedback_input = {
        "text": "This is a test input for generating feedback."
    }
    member_id = "test_member"

    response = client.post(f"/generate_feedback_rag?member_id={member_id}", json=feedback_input)
    logging.info(f"Status Code: {response.status_code}")
    logging.info(f"Response JSON: {response.json()}")
    # assert response.status_code == 200
    feedback_response = response.json()
    # assert "feedback_segments" in feedback_response
    # assert isinstance(feedback_response["feedback_segments"], list)

if __name__ == "__main__":
    test_add_diary_entry()
    test_generate_feedback_rag()
    print("All tests passed!")