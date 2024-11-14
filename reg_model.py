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

# 임베딩 모델
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# FAISS 인덱스 초기화
embedding_dim = 768  # 임베딩 차원 설정
index = faiss.IndexFlatL2(embedding_dim)
vector_db = []  # 임베딩과 텍스트를 저장할 리스트

# FAISS vector store에 추가
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

# 문장 자르기
def preprocess_content(content: str) -> List[str]:
    plain_text_content = markdown_to_text(content)
    sentences = re.split(r'(?<=[.!?])\s+(?=[가-힣A-Za-z])', plain_text_content)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# 연관된 문장 탐색
def search_related_sentences(input_text: str, top_k=3) -> List[str]:
    input_embedding = embedding_model.encode(input_text).astype(np.float32)
    _, indices = index.search(np.array([input_embedding]), top_k)
    related_sentences = [vector_db[idx][0] for idx in indices[0]]
    return related_sentences

# 프롬프트 생성
def generate_prompt(sentence: str, related_context: str) -> str:
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
    return prompt

# 답변 생성
def generate_feedback(prompt: str) -> str:
    feedback_text = ""
    attempt_count = 1
    max_attempts = 5  # 최대 시도 횟수 설정

    while not feedback_text and attempt_count <= max_attempts:
        try:
            logging.info(f"Attempt {attempt_count}: Generating feedback.")
            response = llm(prompt=prompt, max_tokens=150, stop=["\n"])
            feedback_text = response["choices"][0]["text"].strip()
        except Exception as e:
            logging.error(f"Error generating feedback on attempt {attempt_count}: {e}")
            feedback_text = ""
        attempt_count += 1

    if not feedback_text:
        feedback_text = "피드백 생성 실패"
        logging.warning(f"Failed to generate feedback after {max_attempts} attempts.")

    return feedback_text

# ret.txt & vecter DB 갱신 용도
def append_to_registry(sentence: str):
    reg_path = "reg.txt"
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(reg_path, "a", encoding="utf-8") as f:
        f.write(f"[{date_str}] {sentence}\n")

    embedding = embedding_model.encode(sentence).astype(np.float32)
    vector_db.append((sentence, embedding))
    index.add(np.array([embedding], dtype=np.float32))

# chain의 main function
def generate_feedback_with_reg_chain(content: str):
    try:
        logging.info(f"Generating feedback with registry for content: {content}")
        sentences = preprocess_content(content)
        feedback_segments = []
        start_index = 0

        for sentence in sentences:
            related_sentences = search_related_sentences(sentence)
            related_context = " ".join(related_sentences)
            prompt = generate_prompt(sentence, related_context)
            feedback_text = generate_feedback(prompt)

            end_index = start_index + len(sentence)
            feedback_segments.append({
                "startIndex": start_index,
                "endIndex": end_index,
                "feedback": feedback_text
            })

            append_to_registry(sentence)  # ret.txt & vector DB 갱신
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
