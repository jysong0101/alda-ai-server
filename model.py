import re
import logging
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from fastapi import HTTPException
import markdown2

# GGUF 모델을 사용하기 위한 경로 설정
model_name_or_path = "heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF"
model_basename = "ggml-model-Q4_K_M.gguf"

# Hugging Face Hub에서 모델 다운로드
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Llama 모델 로드
llm = Llama(
    model_path=model_path,
    n_threads=8,  # CPU 코어 수
    n_gpu_layers=43,  # GPU에서 실행할 레이어 수
    n_batch=512,  # 배치 크기
    n_ctx=4096  # 컨텍스트 윈도우
)

# 감정 분석 프롬프트
analyze_prompt_template = (
    "You're helpful sentiment analyzer. "
    "Given a sentence, determine which emotion is most dominant. "
    "The emotions are HAPPY (0), SAD (1), ANGERY (2), FEAR (3), SURPRISE (4), DISGUST (5), and NEUTRAL (6). "
    "Answer with only the number corresponding to the emotion. "
    "The response should not contain any other characters. "
    "For each sentence, you HAVE TO list the top THREE dominant emotions in order. "
    "The response format should include a space between each answer (Example: 0 1 2). "
    "You MUST provide EXACTLY three emotions.\n"
    "Human: {sentence}\nAssistant:\n"
)

# 반응 프롬프트
reaction_prompt_template = (
    "Read the provided sentence and give two different responses.\n"
    "First, you are a friend of the person who wrote this sentence. As a friend, give an appropriate response.\n"
    "Second, this person is writing a diary. Encourage them to write more by recommending additional content or asking questions about the previous content, so they continue writing their diary.\n"
    "Clearly separate the two responses as follows. You can replace ().: \n"
    "1. (Friend's response)\n"
    "2. (Diary writing recommendation)\n"
    "Sentence: {sentence}\n"
)

def remove_brackets(text: str) -> str:
    return re.sub(r'\(.*?\):', '', text).strip()

# 감정 분석
def analyze_emotion(sentence: str):
    try:
        logging.info(f"Analyzing sentence: {sentence}")
        prompt = analyze_prompt_template.format(sentence=sentence)

        response = llm(prompt=prompt, max_tokens=50, stop=["\n"])
        output_text = response["choices"][0]["text"]

        logging.info(f"Output text: '{output_text}'")

        response_parts = output_text.split()
        result = {
            "sentence": sentence,
            "predict1": response_parts[0] if len(response_parts) > 0 else "-1",
            "predict2": response_parts[1] if len(response_parts) > 1 else "-1",
            "predict3": response_parts[2] if len(response_parts) > 2 else "-1",
        }

        logging.info(f"Processed sentence: {sentence} | Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error analyzing emotion for sentence '{sentence}': {e}")
        raise HTTPException(status_code=500, detail="Emotion analysis failed.")

# 반응 생성
def generate_reaction(sentence: str):
    try:
        logging.info(f"Generating reaction for sentence: {sentence}")
        prompt = reaction_prompt_template.format(sentence=sentence)

        # 모델에 프롬프트 전달 및 응답 생성
        response = llm(prompt=prompt, max_tokens=300, stop=None)
        output_text = response["choices"][0]["text"]

        logging.info(f"Output text: '{output_text}'")

        # 반응 1과 반응 2로 파싱
        response_parts = output_text.split("1.")
        if len(response_parts) > 1:
            reactions = response_parts[1].split("2.")
            friend_reaction = remove_brackets(reactions[0]) if len(reactions) > 0 else "반응 1 생성 실패"
            diary_reaction = remove_brackets(reactions[1]) if len(reactions) > 1 else "반응 2 생성 실패"
        else:
            friend_reaction = "반응 1 생성 실패"
            diary_reaction = "반응 2 생성 실패"

        result = {
            "sentence": sentence,
            "response1": friend_reaction,
            "response2": diary_reaction
        }

        logging.info(f"Processed sentence: {sentence} | Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error generating reaction for sentence '{sentence}': {e}")
        raise HTTPException(status_code=500, detail="Reaction generation failed.")

def markdown_to_text(content: str) -> str:
    # 마크다운을 HTML로 변환한 뒤 텍스트만 추출
    html = markdown2.markdown(content)
    text = re.sub(r'<[^>]+>', '', html)  # HTML 태그 제거
    text = re.sub(r'\s*\\n\s*', ' ', text)  # 모든 줄바꿈을 마침표와 공백으로 변환
    text = re.sub(r'\s{2,}', ' ', text)  # 여러 개의 공백을 축소
    return text.strip()

def generate_feedback_segments(content: str):
    try:
        logging.info(f"Generating feedback for content: {content}")
        
        # 마크다운을 순수 텍스트로 변환
        plain_text_content = markdown_to_text(content)
        
        logging.info(plain_text_content)
        
        # 문장별로 분리
        sentences = re.split(r'(?<=[.!?])\s+(?=[가-힣A-Za-z])', plain_text_content)
        
        feedback_segments = []
        start_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:  # 빈 문장 무시
                continue

            end_index = start_index + len(sentence)
            
            feedback_text = ""
            attempt_count = 1
            max_attempts = 6

            while not feedback_text and attempt_count < max_attempts:
                try:
                    prompt = (
                        f"Read the provided sentence and respond as if you are a friend of the person who wrote it. "
                        f"Give a short, concise emotional response (1-2 sentences and 100 letters maximum). "
                        f"You Must use language Korean and use following tone: '~해', It's like a real friend."
                        f"When making sentences, always focus on the fact that this is part of a diary. "
                        f"And when calling someone, always call them '친구' or '친구야'."
                        f"Sentence: {sentence}\nFriend's response:"
                    )
                    logging.info(f"Prompt for sentence: {sentence}")
                    logging.info(f"Tring make feedback... (try conut : {attempt_count})")
                    
                    response = llm(prompt=prompt, max_tokens=150, stop=["\n"])
                    feedback_text = response["choices"][0]["text"].strip()
                    
                except Exception as e:
                    logging.error(f"Error generating feedback for sentence '{sentence}': {e}")
                    feedback_text = ""

                attempt_count += 1

            # 시도 후에도 빈 피드백일 경우
            if not feedback_text:
                feedback_text = "피드백 생성 실패"
            
            logging.info(f"Generated feedback for sentence: '{sentence}' -> '{feedback_text}'")  # 생성된 피드백을 로그에 기록
            
            feedback_segments.append({
                "startIndex": start_index,
                "endIndex": end_index,
                "feedback": feedback_text
            })
            
            start_index = end_index + 1

        result = {
            "feedback_segments": feedback_segments
        }

        logging.info(f"Generated feedback result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error generating feedback for content '{content}': {e}")
        raise HTTPException(status_code=500, detail="Feedback generation failed.")
