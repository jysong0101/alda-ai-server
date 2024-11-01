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
        response = requests.post(f"{BASE_URL}/analyze", json={"text": "오늘은 아침부터 상쾌한 기분으로 하루를 시작했어요. 아침 일찍 일어나서 가볍게 스트레칭을 하고, 따뜻한 커피 한 잔과 함께 여유롭게 하루 계획을 세웠죠. 점심에는 친구와 맛있는 음식을 먹으며 즐거운 시간을 보냈고, 오후에는 집중해서 해야 할 일을 모두 마쳤어요. 저녁에는 운동을 하며 하루의 피로를 풀고, 하루를 잘 마무리했다는 뿌듯함을 느꼈답니다. 오늘도 작지만 소중한 일상 속에서 행복을 발견할 수 있어 감사한 하루였어요."})
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
        response = requests.post(f"{BASE_URL}/react", json={"text": "오늘은 아침부터 상쾌한 기분으로 하루를 시작했어요. 아침 일찍 일어나서 가볍게 스트레칭을 하고, 따뜻한 커피 한 잔과 함께 여유롭게 하루 계획을 세웠죠. 점심에는 친구와 맛있는 음식을 먹으며 즐거운 시간을 보냈고, 오후에는 집중해서 해야 할 일을 모두 마쳤어요. 저녁에는 운동을 하며 하루의 피로를 풀고, 하루를 잘 마무리했다는 뿌듯함을 느꼈답니다. 오늘도 작지만 소중한 일상 속에서 행복을 발견할 수 있어 감사한 하루였어요."})
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
