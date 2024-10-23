import logging
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time
from fastapi import HTTPException

# 모델 다운로드 및 경로 설정
model_name_or_path = "heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF"
model_basename = "ggml-model-Q4_K_M.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
logging.info(f"Model downloaded to: {model_path}")

# 모델 로드
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,   # CPU 코어 수
    n_batch=512,   # GPU VRAM에 맞게 조정
    n_gpu_layers=43,  # GPU 레이어 수
    n_ctx=4096,    # 컨텍스트 윈도우 크기
)

logging.info("Model loaded successfully.")

# 감정 분석 프롬프트
analyze_prompt_template = (
    "You're helpful sentiment analyzer. "
    "Given a sentence, determine which emotion is most dominant. "
    "The emotions are fear (0), surprise (1), anger (2), sadness (3), neutral (4), happiness (5), and disgust (6). "
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
    "Sentence: {sentence}\n"
    "Response 1 (As a friend):\n"
    "Response 2 (Diary writing recommendation):\n"
)

# 감정 분석 함수
def analyze_emotion(sentence: str):
    try:
        logging.info(f"Analyzing sentence: {sentence}")
        prompt = analyze_prompt_template.format(sentence=sentence)

        # 모델로 감정 분석 수행
        start = time.time()
        response = lcpp_llm(
            prompt=prompt,
            max_tokens=50,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=['</s>']
        )
        logging.info(f"Model response: {response['choices'][0]['text']}")
        logging.info(f"Time taken: {time.time() - start} seconds")

        result_text = response['choices'][0]['text']
        emotions = result_text.split()

        # 응답 파싱
        if len(emotions) != 3:
            emotions = ["-1", "-1", "-1"]

        result = {
            "Sentence": sentence,
            "Predict 1": emotions[0],
            "Predict 2": emotions[1],
            "Predict 3": emotions[2]
        }
        return result

    except Exception as e:
        logging.error(f"Error analyzing emotion for sentence '{sentence}': {e}")
        raise HTTPException(status_code=500, detail="Emotion analysis failed.")

# 반응 생성 함수
def generate_reaction(sentence: str):
    try:
        logging.info(f"Generating reaction for sentence: {sentence}")
        prompt = reaction_prompt_template.format(sentence=sentence)

        # 모델로 반응 생성 수행
        start = time.time()
        response = lcpp_llm(
            prompt=prompt,
            max_tokens=150,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=['</s>']
        )
        logging.info(f"Model response: {response['choices'][0]['text']}")
        logging.info(f"Time taken: {time.time() - start} seconds")

        result_text = response['choices'][0]['text']

        # '반응 1'과 '반응 2'로 파싱
        response_parts = result_text.split("1.")
        if len(response_parts) > 1:
            reactions = response_parts[1].split("2.")
            friend_reaction = reactions[0].strip() if len(reactions) > 0 else "반응 1 생성 실패"
            diary_reaction = reactions[1].strip() if len(reactions) > 1 else "반응 2 생성 실패"
        else:
            friend_reaction = "반응 1 생성 실패"
            diary_reaction = "반응 2 생성 실패"

        result = {
            "Sentence": sentence,
            "Response 1": friend_reaction,
            "Response 2": diary_reaction
        }
        return result

    except Exception as e:
        logging.error(f"Error generating reaction for sentence '{sentence}': {e}")
        raise HTTPException(status_code=500, detail="Reaction generation failed.")
