import os
import re
import logging
import faiss
from fastapi import HTTPException
import numpy as np
from datetime import datetime
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict

# 모델 및 경로 설정
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

# 임베딩 모델 및 전역 변수 초기화
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
embedding_dim = 768

# 사용자별 벡터 DB와 인덱스 관리 (코사인 유사도용)
user_vector_db: Dict[str, List[Tuple[str, np.ndarray]]] = {}  # 사용자별 (문장, 정규화된 임베딩) 저장
user_indices: Dict[str, faiss.IndexFlatIP] = {}  # 사용자별 FAISS 인덱스 (Inner Product)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    벡터를 정규화하여 크기를 1로 맞춤.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def preprocess_content(content: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[가-힣A-Za-z])', content)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
# 다이어리 데이터를 벡터 DB에 추가
def add_diary_entry_to_db(member_id: str, diary_entry: dict):
    global user_vector_db, user_indices

    # 사용자에 대한 인덱스가 없으면 새로 생성
    if member_id not in user_indices:
        user_indices[member_id] = faiss.IndexFlatIP(embedding_dim)  # 코사인 유사도를 위한 Inner Product 인덱스
        user_vector_db[member_id] = []

    # 일기 내용 처리
    diary_contents = diary_entry["diaryContents"]
    sentences = preprocess_content(diary_contents)  # 문장 나누기

    # 벡터 DB와 RAG 파일에 추가
    for sentence in sentences:
        embedding = embedding_model.encode(sentence).astype(np.float32)
        normalized_embedding = normalize_vector(embedding)  # 정규화
        user_vector_db[member_id].append((sentence, normalized_embedding))
        user_indices[member_id].add(np.array([normalized_embedding], dtype=np.float32))

    # RAG 파일 갱신
    rag_folder = "ragList"
    if not os.path.exists(rag_folder):
        os.makedirs(rag_folder)

    rag_path = os.path.join(rag_folder, f"rag_{member_id}.txt")
    with open(rag_path, "a", encoding="utf-8") as f:
        f.write(f"[{diary_entry['diaryEntryDate']}] {diary_entry['diaryContents']}\n")

    logging.info(f"Added diary entry for member {member_id} to vector DB and RAG.")
# 사용자별 데이터 검색 (코사인 유사도)
def search_related_sentences(member_id: str, input_text: str, top_k=1) -> List[str]:
    """
    사용자(member_id)별 데이터를 기반으로 관련 문장 검색.
    """
    if member_id not in user_indices:
        raise HTTPException(status_code=404, detail=f"No data found for memberId {member_id}")

    input_embedding = embedding_model.encode(input_text).astype(np.float32)
    normalized_input = normalize_vector(input_embedding)  # 입력 벡터 정규화
    _, indices = user_indices[member_id].search(np.array([normalized_input]), top_k)
    related_sentences = [user_vector_db[member_id][idx][0] for idx in indices[0]]
    return related_sentences


# 프롬프트 생성
def generate_prompt(sentence: str, related_context: str) -> str:
    prompt = (
        f"New sentence from diary: {sentence}\n\n"
        f"Related previous entries: {related_context}\n\n"
        f"You are a friendly AI helper who provides reaction on your diary entries."
        f"Considering these entries, respond as a close friend with an emotionally supportive response. "
        f"If there are any related topics or emotions in the previous entries, make sure to incorporate them in your response to provide a more personalized and context-aware reply. "
        f"Generate responses by actively using the relevant sentences provided."
        f"You must use Korean language and the following tone: '~해', as if you're a real friend."
        f"If you ever need to call someone, use the word '친구' or '친구야'."
        f"Respond as if you were a friend, but keep in mind that you are not a real friend."
        f"Think of me as a human being and give me a friendly and proper answer."
        #f"\nSentence: {sentence}\nResponse:"
        f"Response:"
    )
    return prompt

# 답변 생성
def generate_feedback(prompt: str) -> str:
    feedback_text = ""
    attempt_count = 1
    max_attempts = 5

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

def generate_feedback_with_rag_chain(member_id: str, content: str):
    try:
        sentences = preprocess_content(content)
        feedback_segments = []
        start_index = 0

        for sentence in sentences:
            # 관련 문장 검색
            related_sentences = search_related_sentences(member_id, sentence)
            related_context = " ".join(related_sentences)

            # 프롬프트 생성
            prompt = generate_prompt(sentence, related_context)
            
            # 피드백 생성
            feedback_text = generate_feedback(prompt)

            # 피드백 로그에 기록
            logging.info(f"Sentence: {sentence}")
            logging.info(f"Related context: {related_context}")
            logging.info(f"Generated feedback: {feedback_text}")

            # 결과 저장
            end_index = start_index + len(sentence)
            feedback_segments.append({
                "startIndex": start_index,
                "endIndex": end_index,
                "feedback": feedback_text
            })

            start_index = end_index + 1

        result = {"feedback_segments": feedback_segments}

        # 전체 피드백 로그
        logging.info(f"Complete feedback result for memberId {member_id}: {result}")

        return result
    except Exception as e:
        logging.error(f"Error generating feedback with RAG for content '{content}': {e}")
        raise HTTPException(status_code=500, detail="Feedback generation with RAG failed.")
    
def load_rag_files():
    global user_vector_db, user_indices

    rag_folder = "ragList"
    if not os.path.exists(rag_folder):
        logging.warning(f"The folder '{rag_folder}' does not exist. No RAG files to load.")
        return

    rag_files = [f for f in os.listdir(rag_folder) if f.startswith('rag_') and f.endswith('.txt')]

    for rag_file in rag_files:
        member_id = rag_file[len('rag_'):-len('.txt')]
        logging.info(f"Loading RAG file for memberId: {member_id}")

        if member_id not in user_indices:
            user_indices[member_id] = faiss.IndexFlatIP(embedding_dim)  # 코사인 유사도를 위한 Inner Product
            user_vector_db[member_id] = []

        file_path = os.path.join(rag_folder, rag_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
            match = re.match(r'\[(\d{4}-\d{2}-\d{2})\]\s*(.*)', line)
            if match:
                _, diary_contents = match.groups()
                sentences = preprocess_content(diary_contents)
                for sentence in sentences:
                    embedding = embedding_model.encode(sentence).astype(np.float32)
                    normalized_embedding = normalize_vector(embedding)  # 정규화
                    user_vector_db[member_id].append((sentence, normalized_embedding))
                    user_indices[member_id].add(np.array([normalized_embedding], dtype=np.float32))
        logging.info(f"Loaded RAG data for memberId {member_id}: {len(user_vector_db[member_id])} sentences")
