import os
import re
import logging
import faiss
import numpy as np
from datetime import datetime
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from fastapi import HTTPException
import markdown2
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# 모델 초기 설정
model_name_or_path = "heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF"
model_basename = "ggml-model-Q4_K_M.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

llm = Llama(
    model_path=model_path,
    n_threads=8,
    n_gpu_layers=43,
    n_batch=512,
    n_ctx=4096
)

# 임베딩 모델 설정 (예: Sentence-BERT)
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# FAISS 인덱스 초기화
embedding_dim = 768  # 모델의 임베딩 차원 (Sentence-BERT의 경우 768)
index = faiss.IndexFlatL2(embedding_dim)
vector_db = []  # 임베딩과 텍스트를 저장할 리스트

# reg.txt 파일을 불러와 문장 단위로 임베딩하고 벡터 스토어에 저장
def load_registry_to_db():
    reg_path = "reg.txt"
    if not os.path.exists(reg_path):
        logging.warning("Registry file not found.")
        return

    with open(reg_path, "r", encoding="utf-8") as f:
        content = f.read()

    sentences = re.split(r'(?<=[.!?])\s+(?=[가-힣A-Za-z])', content)
    for sentence in sentences:
        embedding = embedding_model.encode(sentence)
        vector_db.append((sentence, embedding))
        index.add(np.array([embedding], dtype=np.float32))  # FAISS 인덱스에 추가

load_registry_to_db()

# 주어진 문장과 유사한 문장을 벡터 스토어에서 검색
def search_related_sentences(input_text: str, top_k=3) -> List[str]:
    input_embedding = embedding_model.encode(input_text).astype(np.float32)
    _, indices = index.search(np.array([input_embedding]), top_k)
    related_sentences = [vector_db[idx][0] for idx in indices[0]]
    return related_sentences

# reg.txt에 문장과 날짜를 추가하는 함수
def append_to_registry(sentence: str):
    reg_path = "reg.txt"
    date_str = datetime.now().strftime("%Y-%m-%d")  # 현재 날짜를 '년-월-일' 형식으로 가져옴
    with open(reg_path, "a", encoding="utf-8") as f:
        f.write(f"[{date_str}] {sentence}\n")

    # 새로운 문장을 벡터 스토어에 추가
    embedding = embedding_model.encode(sentence).astype(np.float32)
    vector_db.append((sentence, embedding))
    index.add(np.array([embedding], dtype=np.float32))  # FAISS 인덱스에 추가

# 문장별 피드백 생성 API에 reg 반영
def generate_feedback_with_reg(content: str):
    try:
        logging.info(f"Generating feedback with registry for content: {content}")

        # 마크다운을 텍스트로 변환 후 문장 분할
        plain_text_content = markdown_to_text(content)
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

            # 관련된 문장 검색
            related_sentences = search_related_sentences(sentence)
            related_context = " ".join(related_sentences)

            while not feedback_text and attempt_count < max_attempts:
                try:
                    prompt = (
                        f"New sentence from diary: {sentence}\n\n"
                        f"Related previous entries: {related_context}\n\n"
                        f"Considering these entries, respond as a close friend with an emotionally supportive response. "
                        f"If there are any related topics or emotions in the previous entries, make sure to incorporate them in your response to provide a more personalized and context-aware reply. "
                        f"You must use Korean language and the following tone: '~해', as if you're a real friend. "
                        f"When creating sentences, always remember this is part of a diary. "
                        f"And when addressing someone, always call them '친구' or '친구야'."
                        f"\nSentence: {sentence}\nFriend's response:"
                    )


                    logging.info(f"Prompt for sentence: {sentence}")
                    logging.info(f"Trying to generate feedback... (attempt count: {attempt_count})")
                    
                    response = llm(prompt=prompt, max_tokens=150, stop=["\n"])
                    feedback_text = response["choices"][0]["text"].strip()
                    
                except Exception as e:
                    logging.error(f"Error generating feedback for sentence '{sentence}': {e}")
                    feedback_text = ""

                attempt_count += 1

            # 시도 후에도 빈 피드백일 경우
            if not feedback_text:
                feedback_text = "피드백 생성 실패"

            logging.info(f"Generated feedback for sentence: '{sentence}' -> '{feedback_text}'")
            
            feedback_segments.append({
                "startIndex": start_index,
                "endIndex": end_index,
                "feedback": feedback_text
            })

            # 생성된 문장을 reg.txt와 벡터 스토어에 추가
            append_to_registry(sentence)
            
            start_index = end_index + 1

        result = {"feedback_segments": feedback_segments}
        logging.info(f"Generated feedback result with registry: {result}")
        return result
    except Exception as e:
        logging.error(f"Error generating feedback with registry for content '{content}': {e}")
        raise HTTPException(status_code=500, detail="Feedback generation with registry failed.")

def markdown_to_text(content: str) -> str:
    html = markdown2.markdown(content)
    text = re.sub(r'<[^>]+>', '', html)
    return re.sub(r'\s{2,}', ' ', text.strip())
