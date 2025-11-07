import os
import glob
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# --- 1. 설정 (Configuration) ---
# 사전 훈련된 모델 이름 또는 경로
PRE_TRAINED_MODEL_NAME = "google/mt5-base"
# 데이터가 있는 폴더 경로
DATA_PATH = "../data"
# 미세조정된 모델이 저장될 경로
OUTPUT_DIR = "./ko-ja-translator"

# --- 2. 데이터 로드 및 전처리 ---
def load_and_prepare_dataset():
    """
    data 폴더의 모든 CSV 파일을 읽어 Hugging Face Dataset 객체로 변환합니다.
    """
    print(f"'{DATA_PATH}' 폴더에서 CSV 파일을 로드합니다...")
    all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"'{DATA_PATH}' 폴더에 CSV 파일이 없습니다. 데이터를 확인해주세요.")

    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # 필요한 열만 선택하고, 누락된 값이 있는 행은 제거
    df = df[["한국어", "일본어"]].dropna()
    
    # 데이터를 훈련 세트와 검증 세트로 분할 (90% 훈련, 10% 검증)
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    print(f"총 {len(df)}개의 샘플 로드 완료.")
    print(f"훈련 데이터: {len(train_df)}개, 검증 데이터: {len(eval_df)}개")

    # Pandas DataFrame을 Hugging Face Dataset으로 변환
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    return train_dataset, eval_dataset

def preprocess_function(examples, tokenizer):
    """
    데이터셋을 토크나이징하는 함수
    """
    inputs = [ex for ex in examples["한국어"]]
    targets = [ex for ex in examples["일본어"]]
    
    # 모델의 입력 형식에 맞게 토크나이징
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # 레이블(정답) 토크나이징
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- 3. 모델 미세조정 ---
def main():
    """
    메인 실행 함수
    """
    # 데이터셋 로드
    train_dataset, eval_dataset = load_and_prepare_dataset()

    # 토크나이저 및 모델 로드
    print(f"'{PRE_TRAINED_MODEL_NAME}' 모델과 토크나이저를 로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # 데이터셋 토크나이징
    print("데이터셋을 토크나이징합니다...")
    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # 데이터 콜레이터 설정 (배치 생성 시 동적 패딩)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 훈련 인자(argument) 설정
    # 이 값들은 필요에 따라 조정할 수 있습니다.
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,               # 모델과 결과물이 저장될 디렉토리
        num_train_epochs=1,                  # 훈련 에폭 수 (전체 데이터셋 반복 횟수)
        per_device_train_batch_size=8,       # 훈련 시 디바이스당 배치 크기=
        per_device_eval_batch_size=8,        # 평가 시 디바이스당 배치 크기
        warmup_steps=500,                    # 학습률을 서서히 증가시키는 초기 스텝 수
        weight_decay=0.01,                   # 가중치 감소 (오버피팅 방지)
        logging_dir="./logs",                # 로그가 저장될 디렉토리
        logging_steps=10,                    # 로그 출력 간격
        eval_steps=500,                      # 검증 데이터셋으로 평가할 스텝 간격
        save_steps=500,                      # 모델 체크포인트를 저장할 스텝 간격
        load_best_model_at_end=False,         # 훈련 종료 시 최적 모델 로드 여부
        predict_with_generate=True,          # 평가 시 텍스트 생성을 함께 수행
        fp16=False,                          # 16비트 부동소수점 정밀도 사용 여부 (GPU에서만 유효)
    )

    # 트레이너 객체 생성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 훈련 시작
    print("모델 미세조정을 시작합니다. 이 작업은 데이터 크기에 따라 몇 시간 이상 걸릴 수 있습니다.")
    trainer.train()

    # 훈련된 모델 저장
    print(f"훈련이 완료되었습니다. 모델을 '{OUTPUT_DIR}'에 저장합니다.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
