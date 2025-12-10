import streamlit as st
import pandas as pd
import os
import re
import json
import shutil
import csv

# --- LangChain 및 Ollama 컴포넌트 ---
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 환경 설정 ---
CHROMA_PATH = "chroma_db"
RECIPES_FILE = "my_saved_recipes.csv"
USER_PROFILE_FILE = "user_profile.json"
RAG_KB_FILE = "recipes.csv" # AI 지식 기반 파일
MODEL_NAME = "mistral" # Mistral 모델 고정

# --- 사용자 프로필 헬퍼 함수 ---
def load_user_profile():
    if os.path.exists(USER_PROFILE_FILE):
        try:
            with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_user_profile(profile_data):
    with open(USER_PROFILE_FILE, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, ensure_ascii=False, indent=4)
    st.toast("냉장고 재료 및 선호도가 저장되었습니다! 💾")


# RAG 구성 요소 초기화 (캐싱)
@st.cache_resource
def setup_rag():
    """RAG 체인 및 구성 요소를 설정합니다."""
    try:
        # 1. 임베딩 및 벡터 스토어 로드
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        # MMR 검색 유지 (다양성 확보)
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 30} 
        )

        # 2. LLM 설정
        llm = ChatOllama(model=MODEL_NAME, temperature=0.01) 

        # 3. RAG 프롬프트 템플릿 정의
        template = """
        당신은 사용자의 냉장고 재료를 기반으로 맞춤형 레시피를 추천하는 AI 셰프입니다.

        ### [핵심 규칙] ###
        1. **알레르기 재료는 레시피에 절대 포함되어서는 안 됩니다. (재료 목록에서 완전히 제외)**
        2. 레시피는 **사용자가 제공한 냉장고 속 재료만** 사용하여 만들 수 있는 **현실적인 일반 레시피**만 추천해야 합니다.
        3. **창작 레시피나 허구의 레시피는 절대 금지**합니다.
        4. **선호 요리**는 냉장고 재료가 **충분할 때만** 고려하고, 재료가 부족하면 선호 요리를 고려하지 않아도 됩니다.
        5. **[가장 중요]** 레시피에 필요한 재료 중 **사용자 냉장고에 없는 재료**가 있다면, AI는 **해당 레시피를 추천 목록에서 완전히 제외**해야 합니다. 오직 **냉장고 재료만**으로 만들 수 있는 레시피만 추천해야 합니다.
        ###################################

        [레시피 선택 논리]:
        - **우선**: 검색된 레시피(Context) 중, **사용자의 핵심 재료**를 가장 많이 사용하는 레시피를 선택합니다.
        - **추천 개수**: 냉장고 재료만으로 조리 가능한 레시피를 **최대 3개**까지 추천합니다.
        
        # [수정된 출력 포맷]
        # 출력 포맷:
        - 모든 응답은 **완벽한 한국어**로 작성되어야 합니다.
        - **각 레시피는 반드시 N. [요리 이름] 형식으로 시작하는 목록 형식**으로 제시하며, 다음 항목을 **정확한 헤더와 함께** 포함해야 합니다:
        
            N. [요리 이름]
            **재료 목록:**
            - (재료 1)
            - (재료 2)
            **상세 조리 과정:**
            - (과정 1)
            - (과정 2)

        **[주의]**
        - '재료 목록:'과 '상세 조리 과정:' 헤더를 반드시 포함해야 합니다.
        - 요리 이름은 **반드시 대괄호 `[]` 안에** 넣어야 합니다.
        - 재료 부족 표시 또는 영양 성분은 절대 출력하지 마세요.


        [사용자 요청]
        {question}

        [관련 레시피 정보 (Context)]
        {context}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 4. RAG 체인 구성
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        st.error(f"RAG 설정 오류: Ollama 데몬이 실행 중이고, '{MODEL_NAME}' 모델이 존재하는지 확인하세요. ({e})")
        return None

# --- 데이터 저장 및 불러오기 함수 ---

def load_recipes():
    """로컬 CSV 파일에서 '나의 레시피' 데이터를 불러옵니다."""
    if os.path.exists(RECIPES_FILE):
        return pd.read_csv(RECIPES_FILE, encoding='utf-8')
    
    return pd.DataFrame(columns=['이름', '생성일', '재료_요약', '전체_레시피'])

# --- recipes.csv에 레시피 추가 로직 ---
def append_to_rag_kb(name, materials, allergies, preference, process):
    """
    (수동 추가) 레시피를 recipes.csv (RAG 지식 기반)에 추가합니다.
    """
    
    materials_single_line = materials.replace('\n', ' ').replace('\r', ' ')
    process_single_line = process.replace('\n', ' ').replace('\r', ' ')

    new_row = [
        name,
        materials_single_line,
        allergies if allergies else '없음',
        preference if preference else '기타',
        process_single_line
    ]
    
    file_exists = os.path.exists(RAG_KB_FILE)
    
    # newline='' 옵션은 csv 모듈 사용 시 필수
    with open(RAG_KB_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 파일이 비어있거나 새로 생성된 경우 헤더 작성
        if not file_exists or os.path.getsize(RAG_KB_FILE) == 0:
            header = ['이름', '재료', '알레르기', '선호_요리', '레시피_과정']
            writer.writerow(header)
            
        writer.writerow(new_row)


# --- [수정됨] 저장된 레시피 삭제 (헤더 인식 방식 개선) ---
def delete_recipe(recipe_name_to_delete):
    """
    지정된 이름의 레시피를 my_saved_recipes.csv (보기 목록)과
    recipes.csv (AI 지식 기반) *모두*에서 삭제합니다.
    (강력한 삭제 모드: 헤더가 깨져도 첫 번째 열을 기준으로 삭제 시도)
    """
    
    if recipe_name_to_delete:
        recipe_name_to_delete = recipe_name_to_delete.strip()
    else:
        return False

    deleted_from_list = False
    deleted_from_kb = False

    # --- 1. my_saved_recipes.csv (보기 목록)에서 삭제 ---
    if os.path.exists(RECIPES_FILE):
        try:
            df_list = pd.read_csv(RECIPES_FILE, encoding='utf-8')
            # 이름 컬럼 정규화 후 비교
            df_list_updated = df_list[df_list['이름'].astype(str).str.strip() != recipe_name_to_delete]
            
            if len(df_list_updated) < len(df_list):
                df_list_updated.to_csv(RECIPES_FILE, index=False, encoding='utf-8')
                deleted_from_list = True
        except Exception as e:
            st.error(f"{RECIPES_FILE} 삭제 중 오류: {e}")

    # --- 2. recipes.csv (AI 지식 기반)에서 삭제 [강력한 삭제 로직 적용] ---
    if os.path.exists(RAG_KB_FILE):
        try:
            # 1차 시도: 헤더가 있다고 가정하고 읽기
            try:
                df_kb = pd.read_csv(RAG_KB_FILE, encoding='utf-8')
                df_kb.columns = df_kb.columns.str.strip() # 컬럼명 공백 제거
            except:
                # 읽기 실패 시 헤더 없이 읽기
                df_kb = pd.read_csv(RAG_KB_FILE, encoding='utf-8', header=None)

            original_len = len(df_kb)
            
            # '이름' 컬럼이 존재하는지 확인
            if '이름' in df_kb.columns:
                # '이름' 컬럼 기준으로 삭제
                df_kb = df_kb[df_kb['이름'].astype(str).str.strip() != recipe_name_to_delete]
            else:
                # '이름' 컬럼이 없으면 무조건 첫 번째 열(index 0)을 기준으로 삭제
                # (데이터가 꼬였을 때를 대비한 안전장치)
                df_kb = df_kb[df_kb.iloc[:, 0].astype(str).str.strip() != recipe_name_to_delete]

            # 삭제된 내용이 있다면 파일 저장
            if len(df_kb) < original_len:
                # 저장할 때 포맷을 깔끔하게 정리 (utf-8-sig는 엑셀 호환성용)
                df_kb.to_csv(RAG_KB_FILE, index=False, encoding='utf-8-sig')
                deleted_from_kb = True
                
        except Exception as e:
            st.error(f"{RAG_KB_FILE} 삭제 처리 중 치명적 오류: {e}")

    # --- 3. 결과 반환 ---
    if deleted_from_list or deleted_from_kb:
        st.toast(f"'{recipe_name_to_delete}' 레시피가 삭제되었습니다. 🗑️")
        return True
    else:
        st.warning(f"'{recipe_name_to_delete}' 레시피를 파일에서 찾을 수 없습니다. (이미 삭제되었거나 이름이 다를 수 있습니다.)")
        return False


# --- save_recipe (전체 텍스트 저장 전용) ---
def save_recipe(recipe_text, name):
    """
    AI 응답 텍스트 '전체'를 my_saved_recipes.csv에 저장합니다.
    """
    
    summary = (recipe_text[:70] + '...') if len(recipe_text) > 70 else recipe_text
    summary = summary.replace('\n', ' ').replace('*', '')

    data = {
        '이름': name if name else 'AI 추천 레시피',
        '생성일': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        '재료_요약': summary,
        '전체_레시피': recipe_text
    }
    
    df_new = pd.DataFrame([data])
    
    if os.path.exists(RECIPES_FILE):
        df_existing = load_recipes()
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
        
    df.to_csv(RECIPES_FILE, index=False, encoding='utf-8')
    
    st.success(f"'{name}' 레시피 묶음이 저장되었습니다! 💾")


# --- 수동 추가 레시피를 '나의 레시피' 목록에도 저장 ---
def save_manual_recipe_to_list(name, materials, process):
    """
    '지식 기반 추가' 탭에서 입력한 레시피를 my_saved_recipes.csv (보여지는 목록)에도 추가합니다.
    """
    
    full_content_for_storage = f"**{name}**\n\n**재료 목록:**\n{materials}\n\n**상세 조리 과정:**\n{process}"
    
    summary = (materials[:70] + '...') if len(materials) > 70 else materials
    summary = summary.replace('\n', ', ')

    data = {
        '이름': name,
        '생성일': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        '재료_요약': summary,
        '전체_레시피': full_content_for_storage
    }

    df_new = pd.DataFrame([data])
    
    if os.path.exists(RECIPES_FILE):
        df_existing = load_recipes()
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
        
    df.to_csv(RECIPES_FILE, index=False, encoding='utf-8')
    st.success(f"'{name}' 레시피가 '저장된 레시피 보기' 목록에도 추가되었습니다! 📚")


# --- Streamlit 메인 앱 ---
def main():
    st.set_page_config(page_title="냉장고 해결사 🤖", layout="wide")
    st.title("👨‍🍳 냉장고 해결사: 나만의 요리사 AI")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'ai_response' not in st.session_state:
        st.session_state['ai_response'] = ""
    if 'current_recipe_name' not in st.session_state:
        st.session_state['current_recipe_name'] = ""
    if 'user_profile' not in st.session_state:
        st.session_state['user_profile'] = load_user_profile()

    category = st.sidebar.radio(
        "카테고리", 
        ["메인", "나의 냉장고", "레시피 생성", "나의 레시피"]
    )
    
    rag_chain = setup_rag()
    
    if category == "메인":
        st.subheader("🎉 냉장고 해결사에 오신 것을 환영합니다!")
        st.markdown("""
        이 앱은 여러분의 냉장고 속 재료를 기반으로 맞춤형 레시피를 추천해주는 AI 셰프입니다.
        AI는 여러분이 가진 재료만으로 만들 수 있는 **현실적인 레시피**를 제안합니다.

        **주요 기능:**

        * **[나의 냉장고]**: 현재 내가 가지고 있는 재료와 알레르기 정보, 선호하는 요리 스타일을 로컬로 저장합니다.
        * **[레시피 생성]**: '나의 냉장고' 정보를 바탕으로 AI가 만들 수 있는 레시피를 추천해줍니다. 생성된 레시피는 저장할 수 있습니다.
        * **[나의 레시피]**: 저장된 레시피를 관리할 수 있습니다. 직접 레시피를 작성해 AI에게 학습시킬 수 있습니다.

        ---
        *AI 지식 기반(RAG)은 `recipes.csv` 파일을 기반으로 합니다.*
        *AI 모델은 `mistral` (Ollama)을 사용합니다.*
        """)

    elif category == "나의 냉장고":
        st.subheader("🧊 나의 냉장고 및 선호도 설정")
        
        with st.form("ingredients_form"):
            default_ingredients = st.session_state['user_profile'].get('ingredients', '')
            ingredients_input = st.text_area(
                "재료 목록을 입력하세요 (재료, 양, 쉼표로 구분):",
                value=default_ingredients,
                key='profile_ingredients'
            )
            
            default_preferences = st.session_state['user_profile'].get('preferences', '')
            preferences_input = st.text_area(
                "선호 요리 종류, 알레르기 등 추가 조건:",
                value=default_preferences,
                key='profile_preferences'
            )
            
            submitted = st.form_submit_button("냉장고 정보 저장")
            
            if submitted:
                new_profile = {
                    'ingredients': ingredients_input,
                    'preferences': preferences_input
                }
                st.session_state['user_profile'] = new_profile
                save_user_profile(new_profile)

    elif category == "레시피 생성":
        
        st.subheader("💬 AI 챗봇 레시피 추천")
        
        if rag_chain is None:
            return 

        with st.form("recipe_form"):
            default_ingredients = st.session_state['user_profile'].get('ingredients', '')
            default_preferences = st.session_state['user_profile'].get('preferences', '')
            
            ingredients = st.text_input(
                "📦 냉장고 속 재료와 양을 입력하세요:", 
                value=default_ingredients, 
                key='runtime_ingredients'
            )
            preferences = st.text_area(
                "🌟 선호 요리 종류, 알레르기 등 추가 조건을 입력하세요:", 
                value=default_preferences, 
                key='runtime_preferences'
            )
            submitted = st.form_submit_button("레시피 생성")

        if submitted and ingredients:
            user_query = f"재료: {ingredients}. 추가 조건: {preferences}"
            
            with st.spinner(f"AI 셰프({MODEL_NAME})가 레시피를 생성 중입니다..."):
                try:
                    ai_response = rag_chain.invoke(user_query)
                    st.session_state['ai_response'] = ai_response
                except Exception as e:
                    st.error(f"레시피 생성 중 오류 발생: {e}")
                    st.session_state['ai_response'] = ""
                    
        if st.session_state['ai_response']:
            st.success("✅ 레시피가 완성되었습니다!")
            
            st.markdown(st.session_state['ai_response'])
            st.markdown("---")
            
            default_name = f"AI 추천 묶음 ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})"
            
            if st.button("이 추천 레시피 묶음 저장 💾", key='save_all_btn'):
                save_recipe(st.session_state['ai_response'], default_name)

    elif category == "나의 레시피":
        tab1, tab2 = st.tabs(["📚 저장된 레시피 보기", "📝 지식 기반 추가 (RAG)"])
        
        with tab1:
            st.subheader("📚 저장된 나의 레시피 목록")
        
            df_recipes = load_recipes()
            
            if df_recipes.empty:
                st.info("아직 저장된 레시피가 없습니다. 레시피를 생성하고 저장해보세요.")
            else:
                display_cols = ['이름', '생성일', '재료_요약']
                st.dataframe(df_recipes[display_cols], width='stretch') # use_container_width 대체

                st.markdown("---")
                st.caption("레시피를 선택하고 삭제하려면 아래를 확인하세요.")
                
                recipe_names = df_recipes['이름'].tolist()
                selected_name = st.selectbox("상세 레시피 선택 및 삭제:", ["선택하세요"] + recipe_names, key='recipe_select')
                
                if selected_name != "선택하세요":
                    selected_recipe_data = df_recipes[df_recipes['이름'] == selected_name]
                    
                    if not selected_recipe_data.empty:
                        selected_recipe = selected_recipe_data.iloc[0]
                        
                        st.markdown(f"### {selected_recipe['이름']}")
                        st.markdown(f"**생성일:** {selected_recipe['생성일']}")
                        st.markdown("---")
                        st.markdown("#### 전체 레시피 및 분석 내용")
                        st.markdown(selected_recipe['전체_레시피'])

                        if st.button(f"'{selected_name}' 레시피 삭제 🗑️", key='delete_saved_recipe'):
                            # 삭제가 성공하면 페이지 리런
                            if delete_recipe(selected_name):
                                st.rerun()
                    else:
                        st.warning("레시피를 찾는 중 오류가 발생했습니다. 페이지를 새로고침하세요.")


        with tab2:
            st.subheader("📝 새 레시피 추가 (AI 지식 기반 확장)")
            st.info("여기에 추가하는 레시피는 AI가 추천 레시피를 생성할 때 참조하는 데이터베이스에 추가됩니다. **(저장 후 RAG 재설정 필수)**")
            
            with st.form("new_recipe_form"):
                new_name_input = st.text_input("요리 이름", key="new_name")
                new_materials = st.text_area("재료 목록 (예: 양파 1개, 소고기 200g, 쉼표로 구분)", key="new_materials")
                new_allergies = st.text_input("알레르기 재료 (레시피가 포함하는 알레르기 유발 재료. 없으면 공백)", key="new_allergies")
                new_preference = st.text_input("선호 요리 종류 (예: 한식, 양식)", key="new_preference")
                new_process = st.text_area("상세 조리 과정", key="new_process")
                
                submitted_new = st.form_submit_button("지식 기반에 레시피 추가")
                
                if submitted_new and new_name_input and new_materials and new_process:
                    new_name = new_name_input.strip()

                    if not new_name:
                        st.error("요리 이름은 공백일 수 없습니다.")
                    else:
                        # 1. RAG KB(recipes.csv)에 저장
                        append_to_rag_kb(
                            new_name, 
                            new_materials, 
                            new_allergies, 
                            new_preference, 
                            new_process
                        )
                        
                        # 2. '저장된 레시피 보기' 목록에도 저장
                        save_manual_recipe_to_list(new_name, new_materials, new_process)
                        
                        st.success(f"'{new_name}' 레시피가 지식 기반(RAG)에 추가되었습니다!")
                        st.warning("⚠️ **새로운 레시피를 AI가 인식하도록 RAG 데이터베이스를 재설정해야 합니다.**")
                        st.code("./venv/Scripts/python.exe rag_setup.py", language='bash')
                        st.info("터미널에서 위 명령을 실행해주세요.")


if __name__ == "__main__":
    main()