# app.py

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropicMessages

# ✅ 환경 변수 로드
load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("❌ .env 파일에 ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# ✅ 상수 설정
ROOT_DIR = "C:/KDT13/kh0616/project_12/hyundaicar_info"
PERSIST_DIR = "vector_store_index"

# ✅ 전체 PDF 자동 로딩 → 벡터스토어 생성 함수
def load_all_car_pdfs_to_vectorstore(root_path: str, persist_dir: str = "vector_store_index") -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = []

    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue

    # 차량 모델별 폴더 탐색
        for car_model in os.listdir(category_path):
            car_model_path = os.path.join(category_path, car_model)
            if not os.path.isdir(car_model_path):
                continue

            for filename in os.listdir(car_model_path):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(car_model_path, filename)
                    try:
                        docs = PyPDFLoader(pdf_path).load()
                        chunks = splitter.split_documents(docs)
                        all_docs.extend(chunks)
                        print(f"✅ 로딩 완료: {pdf_path}")
                    except Exception as e:
                        print(f"❌ 로딩 실패: {pdf_path} ({e})")

    st.success(f"📦 총 문서 수: {len(all_docs)}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(all_docs, embedding=embeddings, persist_directory=persist_dir)
    return vectordb

# ✅ Streamlit UI 시작
st.set_page_config(page_title="🚗 현대차 Claude RAG 데모", layout="wide")
st.title("🚗 현대차 Claude 기반 RAG 데모")
st.caption("폴더에 저장된 PDF 문서를 자동으로 읽어 Claude 기반 답변 생성")

# ✅ 캐시된 벡터스토어 생성
@st.cache_resource
def get_vectordb():
    return load_all_car_pdfs_to_vectorstore(ROOT_DIR, PERSIST_DIR)

vectordb = get_vectordb()

# ✅ 사용자 질문 입력
question = st.text_input("❓ 현대차 차량에 대해 궁금한 점을 입력하세요 (예: '스타리아의 적재 용량은?')")

# ✅ 질문 처리
if question:
    with st.spinner("🔍 Claude 기반 응답 생성 중..."):
        retriever = vectordb.as_retriever()
        llm = ChatAnthropicMessages(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.run(question)

    # ✅ 결과 출력
    st.markdown("### 💡 Claude 기반 PDF 응답")
    st.write(response)
