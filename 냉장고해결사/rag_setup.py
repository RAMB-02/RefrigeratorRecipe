# rag_setup.py

import os
import shutil # <--- 폴더 삭제를 위해 추가
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import pandas as pd

# --- 환경 설정 ---
FILE_PATH = "recipes.csv"
CHROMA_PATH = "chroma_db"
MODEL_NAME = "mistral" # app.py와 동일하게 mistral 사용
# Windows CSV 파일 인코딩 문제 해결 (cp949 또는 latin-1)
ENCODING = "utf-8" 

def create_vector_store():
    """RAG 벡터 스토어를 생성하고 ChromaDB에 저장합니다."""
    
    # 1. 파일 존재 확인
    if not os.path.exists(FILE_PATH):
        print(f"오류: {FILE_PATH} 파일이 존재하지 않습니다. AI 지식 기반 파일을 생성해주세요.")
        return

    # 2. [수정] 기존 chroma_db 폴더 강제 삭제
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print(f"기존 {CHROMA_PATH} 폴더를 삭제했습니다. (강제 재설정)")
        except Exception as e:
            print(f"경고: {CHROMA_PATH} 삭제 실패 ({e}). 계속 진행합니다.")

    # 3. CSV 파일 로드 (인코딩 수정 반영)
    try:
        loader = CSVLoader(
            file_path=FILE_PATH, 
            encoding=ENCODING,  # <--- 인코딩 문제 해결 (cp949 또는 latin-1)
            csv_args={'delimiter': ','}
        )
        documents = loader.load()
    except Exception as e:
        print(f"CSV 파일 로드 중 심각한 인코딩 오류 발생 ({e}).")
        print(f"인코딩이 '{ENCODING}'이 아닐 수 있습니다. 'latin-1' 또는 'utf-8'로 변경해보세요.")
        return

    # 4. Ollama 임베딩 모델 설정
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
    except Exception as e:
        print(f"Ollama 임베딩 로드 오류: {e}")
        print(f"'{MODEL_NAME}' 모델이 Ollama에 다운로드되어 실행 가능한지 확인하세요.")
        return

    # 5. 문서를 벡터화하여 ChromaDB에 저장
    print("\nVector Store 생성 중...")
    
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Vector Store 생성 완료! 저장 위치: {CHROMA_PATH}")
    return vector_store

if __name__ == "__main__":
    create_vector_store()