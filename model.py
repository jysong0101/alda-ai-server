import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator
from fastapi import HTTPException

accelerator = Accelerator()

model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = accelerator.prepare(model)
model.eval()
logging.info("Model and tokenizer loaded successfully.")

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

# 감정 분석
def analyze_emotion(sentence: str):
    try:
        logging.info(f"sentence : {sentence}")
        encoded_sentence = sentence.encode('utf-8').decode('utf-8')
        result = {'Sentence': encoded_sentence}
        
        model_inputs = tokenizer(analyze_prompt_template.format(sentence=encoded_sentence), return_tensors='pt')
        model_inputs = model_inputs.to(accelerator.device)

        with torch.no_grad():
            outputs = model.generate(**model_inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = output_text.split("\n")[-1].split(" ")

        logging.info(f"output_text : '{output_text}'")

        response = [r for r in response if r.isdigit()]
        if len(response) == 1:
            response.extend(["-1", "-1"])
        elif len(response) == 2:
            response.append("-1")

        if len(response) != 3:
            logging.warning(f"Invalid response: {response}")
            response = ["-1", "-1", "-1"]

        result['Predict 1'] = response[0]
        result['Predict 2'] = response[1]
        result['Predict 3'] = response[2]

        logging.info(f"Processed sentence: {encoded_sentence} | Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error analyzing emotion for sentence '{sentence}': {e}")
        raise HTTPException(status_code=500, detail="Emotion analysis failed.")

# 반응
def generate_reaction(sentence: str):
    try:
        logging.info(f"sentence : {sentence}")
        encoded_sentence = sentence.encode('utf-8').decode('utf-8')
        result = {'Sentence': encoded_sentence}

        model_inputs = tokenizer(reaction_prompt_template.format(sentence=encoded_sentence), return_tensors='pt')
        model_inputs = model_inputs.to(accelerator.device)

        with torch.no_grad():
            outputs = model.generate(**model_inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        logging.info(f"output_text : '{output_text}'")

        # '반응 1'과 '반응 2'로 파싱
        response_parts = output_text.split("1.")
        if len(response_parts) > 1:
            reactions = response_parts[1].split("2.")
            friend_reaction = reactions[0].strip() if len(reactions) > 0 else "반응 1 생성 실패"
            diary_reaction = reactions[1].strip() if len(reactions) > 1 else "반응 2 생성 실패"
        else:
            friend_reaction = "반응 1 생성 실패"
            diary_reaction = "반응 2 생성 실패"

        result['Response 1'] = friend_reaction
        result['Response 2'] = diary_reaction

        logging.info(f"Processed sentence: {encoded_sentence} | Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error generating reaction for sentence '{sentence}': {e}")
        raise HTTPException(status_code=500, detail="Reaction generation failed.")
