import re
import logging
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from fastapi import HTTPException

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

# 괄호 안의 텍스트 제거 함수
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
