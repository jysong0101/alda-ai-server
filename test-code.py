import requests
import logging
from datetime import datetime


# 로그 파일 설정 (날짜 및 시간 정보를 포함)
log_filename = f"test-logs/test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 서버 URL
BASE_URL = "http://127.0.0.1:8000"

def test_analyze():
    try:
        logging.info("Starting Analyze Test.")
        
        # 테스트 요청 보내기
        response = requests.post(f"{BASE_URL}/analyze", json={"text": "오늘 하루는 정말 힘들었어."})
        logging.info(f"Request to /analyze sent. Status code: {response.status_code}")
        
        # 상태 코드 검증
        assert response.status_code == 200, f"Status code: {response.status_code}. Expected: 200."
        
        # 응답 데이터 확인
        data = response.json()
        assert "Sentence" in data, "'Sentence' field missing."
        assert "Predict 1" in data, "'Predict 1' field missing."
        assert "Predict 2" in data, "'Predict 2' field missing."
        assert "Predict 3" in data, "'Predict 3' field missing."

        logging.info(f"Analyze Test Passed. Response: {data}")
        print("Analyze Test Passed")
        
    except Exception as e:
        logging.error(f"Analyze Test Failed: {e}")
        print(f"Analyze Test Failed: {e}")

def test_react():
    try:
        logging.info("Starting React Test.")
        
        # 테스트 요청 보내기
        response = requests.post(f"{BASE_URL}/react", json={"text": "오늘 하루는 정말 힘들었어."})
        logging.info(f"Request to /react sent. Status code: {response.status_code}")
        
        # 상태 코드 검증
        assert response.status_code == 200, f"Status code: {response.status_code}. Expected: 200."
        
        # 응답 데이터 확인
        data = response.json()
        assert "Sentence" in data, "'Sentence' field missing."
        assert "Response 1" in data, "'Response 1' field missing."
        assert "Response 2" in data, "'Response 2' field missing."
        
        logging.info(f"React Test Passed. Response: {data}")
        print("React Test Passed")
        
    except Exception as e:
        logging.error(f"React Test Failed: {e}")
        print(f"React Test Failed: {e}")

if __name__ == "__main__":
    test_analyze()
    test_react()
