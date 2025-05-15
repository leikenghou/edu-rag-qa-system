import os
import logging
import hashlib
import shutil
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import SentenceTransformer

# ==================== 配置常量 ====================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_SEARCH_K = 5
BM25_SEARCH_K = 5
DEFAULT_VECTOR_WEIGHT = 0.5
PAGE_SIZE = 5
VECTOR_STORE_DIR = "./chroma_db"  # 修改为更明确的目录名

# ==================== 初始化设置 ====================
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 绕过 Streamlit/PyTorch 错误
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# 加载环境变量
load_dotenv()
deepseek_api_key = os.getenv("API_KEY")
base_url = "https://api.deepseek.com"


# ==================== Session State 初始化 ====================
def initialize_session_state():
    """确保所有需要的 session state 变量都已初始化"""
    required_states = {
        "processed_files": set(),
        "all_docs": [],
        "bm25_retriever": None,
        "ensemble_retriever": None,
        "vector_weight": DEFAULT_VECTOR_WEIGHT,
        "vectorstore": None,
        "embeddings": None,
        "retrieval_mode": "混合检索",
        "last_refresh_time": None,
        "current_page": 1,
        "last_question": "",
        "last_answer": "",
        "last_context": []
    }

    for key, default_value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# 初始化session state
initialize_session_state()


# ==================== 模型初始化 ====================
@st.cache_resource
def initialize_models():
    """初始化语言模型和嵌入模型"""
    try:
        # 初始化语言模型
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=deepseek_api_key,
            base_url=base_url,
            temperature=0.7
        )

        # 初始化嵌入模型
        model_name = "BAAI/bge-m3"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        return llm, embeddings
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        st.error(f"模型初始化失败: {str(e)}")
        st.stop()


# ==================== 向量数据库初始化 ====================
@st.cache_resource
def initialize_vectorstore(_embeddings, persist_directory=VECTOR_STORE_DIR):
    """初始化或重建向量数据库"""
    try:
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)

        # 检查是否已有数据库
        collection_files = [
            "chroma-collections.parquet",
            "chroma-embeddings.parquet"
        ]
        db_exists = all(os.path.exists(os.path.join(persist_directory, f)) for f in collection_files)

        # 初始化向量数据库
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=_embeddings
        )

        # 初始化session state中的文档列表
        if len(st.session_state.all_docs) == 0 and db_exists:
            try:
                # 从数据库加载已有文档
                db_content = vectorstore.get()
                if db_content and "documents" in db_content and len(db_content["documents"]) > 0:
                    st.session_state.all_docs = [
                        Document(
                            page_content=doc,
                            metadata=db_content["metadatas"][i] if "metadatas" in db_content else {}
                        )
                        for i, doc in enumerate(db_content["documents"])
                    ]
                    logger.info(f"从数据库加载了 {len(st.session_state.all_docs)} 个文档片段")

                    # 初始化已处理文件集合
                    for doc in st.session_state.all_docs:
                        if "source" in doc.metadata:
                            file_hash = hashlib.md5(doc.metadata["source"].encode()).hexdigest()
                            st.session_state.processed_files.add(file_hash)
            except Exception as e:
                logger.warning(f"加载已有文档失败: {str(e)}")

        return vectorstore
    except Exception as e:
        logger.error(f"向量数据库初始化失败: {str(e)}")
        st.error(f"向量数据库初始化失败: {str(e)}")
        st.stop()


# 初始化所有组件
try:
    llm, embeddings = initialize_models()
    vectorstore = initialize_vectorstore(embeddings)
    st.session_state.vectorstore = vectorstore
    st.session_state.embeddings = embeddings

    # 初始化BM25检索器
    if len(st.session_state.all_docs) > 0:
        st.session_state.bm25_retriever = BM25Retriever.from_documents(
            st.session_state.all_docs
        )
        st.session_state.bm25_retriever.k = BM25_SEARCH_K
except Exception as e:
    st.error("系统初始化失败，请检查错误信息并刷新页面")
    st.stop()

# ==================== 检索系统设置 ====================
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K})

# 提示模板
prompt_template = """
你是一个知识问答助手。根据以下检索到的上下文回答用户的问题，回答要简洁、准确。如果上下文不足以回答，说明无法回答并建议用户提供更多信息。接着使用deepseek的知识进行回答（需要明确指出）。

**上下文**：
{context}

**问题**：
{question}

**回答**：
"""
prompt = PromptTemplate.from_template(prompt_template)


# ==================== 检索器逻辑 ====================
def get_retriever(_=None):
    """根据当前设置返回适当的检索器"""
    initialize_session_state()

    retrieval_mode = st.session_state.retrieval_mode
    vector_weight = st.session_state.vector_weight

    if retrieval_mode == "仅向量检索":
        return vector_retriever

    if retrieval_mode == "仅关键字检索(BM25)":
        if st.session_state.bm25_retriever is None and st.session_state.all_docs:
            try:
                st.session_state.bm25_retriever = BM25Retriever.from_documents(
                    st.session_state.all_docs
                )
                st.session_state.bm25_retriever.k = BM25_SEARCH_K
            except Exception as e:
                logger.error(f"创建BM25检索器失败: {str(e)}")
                return vector_retriever
        return st.session_state.bm25_retriever or vector_retriever

    # 混合检索模式
    if st.session_state.bm25_retriever is None and st.session_state.all_docs:
        try:
            st.session_state.bm25_retriever = BM25Retriever.from_documents(
                st.session_state.all_docs
            )
            st.session_state.bm25_retriever.k = BM25_SEARCH_K
        except Exception as e:
            logger.error(f"创建BM25检索器失败: {str(e)}")
            return vector_retriever

    if st.session_state.bm25_retriever:
        return EnsembleRetriever(
            retrievers=[vector_retriever, st.session_state.bm25_retriever],
            weights=[vector_weight, 1.0 - vector_weight]
        )
    return vector_retriever


# ==================== RAG 链 ====================
def format_docs(docs):
    """将文档列表格式化为字符串"""
    return "\n".join(doc.page_content for doc in docs)

# 修改后的RAG链
rag_chain = (
    {
        "context": RunnableLambda(get_retriever) | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==================== PDF 处理函数 ====================
def process_pdf_file(uploaded_file):
    """处理上传的 PDF 文件并更新知识库"""
    tmp_file_path = None
    try:
        # 计算文件哈希
        file_content = uploaded_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)

        # 检查是否已处理
        if file_hash in st.session_state.processed_files:
            st.warning("此 PDF 文件已上传并处理，跳过重复嵌入。")
            return

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        # 加载和验证 PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("PDF 文件内容为空或不可读！")

        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)

        # 为每个片段添加源文件信息
        for split in splits:
            split.metadata["source"] = uploaded_file.name

        # 更新向量数据库 (Chroma会自动持久化)
        st.session_state.vectorstore.add_documents(splits)

        # 更新内存中的文档列表
        st.session_state.all_docs.extend(splits)
        st.session_state.processed_files.add(file_hash)

        # 重建 BM25 检索器
        try:
            st.session_state.bm25_retriever = BM25Retriever.from_documents(
                st.session_state.all_docs
            )
            st.session_state.bm25_retriever.k = BM25_SEARCH_K
        except Exception as e:
            logger.error(f"更新BM25检索器失败: {str(e)}")

        st.session_state.last_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"成功处理 {uploaded_file.name}！新增 {len(splits)} 个文档片段。")
    except Exception as e:
        logger.error(f"处理 PDF 文件时出错: {str(e)}")
        st.error(f"处理 PDF 文件时出错: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


# ==================== 侧边栏知识库查看 ====================
def knowledge_base_sidebar():
    """侧边栏知识库查看功能"""
    with st.sidebar:
        st.header("📚 知识库管理")

        # 独立刷新按钮
        if st.button("🔄 刷新知识库", use_container_width=True, help="从磁盘重新加载所有内容"):
            try:
                # 重新初始化向量数据库
                st.session_state.vectorstore = Chroma(
                    persist_directory=VECTOR_STORE_DIR,
                    embedding_function=st.session_state.embeddings
                )

                # 重新加载文档
                db_content = st.session_state.vectorstore.get()
                st.session_state.all_docs = [
                    Document(
                        page_content=doc,
                        metadata=db_content["metadatas"][i] if "metadatas" in db_content else {}
                    )
                    for i, doc in enumerate(db_content["documents"])
                ] if "documents" in db_content else []

                # 重建已处理文件集合
                st.session_state.processed_files = set()
                for doc in st.session_state.all_docs:
                    if "source" in doc.metadata:
                        file_hash = hashlib.md5(doc.metadata["source"].encode()).hexdigest()
                        st.session_state.processed_files.add(file_hash)

                # 重建BM25检索器
                if len(st.session_state.all_docs) > 0:
                    st.session_state.bm25_retriever = BM25Retriever.from_documents(
                        st.session_state.all_docs
                    )
                    st.session_state.bm25_retriever.k = BM25_SEARCH_K

                st.session_state.last_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()
            except Exception as e:
                st.error(f"刷新知识库失败: {str(e)}")

        if st.session_state.last_refresh_time:
            st.caption(f"最后刷新: {st.session_state.last_refresh_time}")

        # 知识库统计信息
        st.divider()
        st.markdown(f"**文档片段总数**: {len(st.session_state.all_docs)}")
        st.markdown(f"**已处理文件数**: {len(st.session_state.processed_files)}")

        # 分页控制
        st.divider()
        total_pages = max(1, (len(st.session_state.all_docs) + PAGE_SIZE - 1) // PAGE_SIZE)
        st.session_state.current_page = st.number_input(
            "页码",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page,
            key="kb_page_input"
        )

        # 文档显示
        st.divider()
        if not st.session_state.all_docs:
            st.info("知识库为空，请上传PDF文件")
            return

        start_idx = (st.session_state.current_page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(st.session_state.all_docs))

        for i in range(start_idx, end_idx):
            doc = st.session_state.all_docs[i]
            with st.expander(f"📄 片段 {i + 1}", expanded=False):
                st.markdown(f"**来源**: `{doc.metadata.get('source', '未知')}`")
                st.markdown(f"**页码**: {doc.metadata.get('page', '未知')}")
                st.markdown("**内容预览**:")
                st.text(doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else ""))
                if st.checkbox("显示完整内容", key=f"full_{i}"):
                    st.text_area("内容", doc.page_content, height=200, key=f"content_{i}", label_visibility="collapsed")


# ==================== 主界面 ====================
def main_interface():
    """主界面功能"""
    st.title("🧠 RAG 知识问答系统")
    st.caption("Powered by DeepSeek & LangChain")

    # 检索设置
    with st.expander("⚙️ 检索配置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "检索模式",
                ["混合检索", "仅向量检索", "仅关键字检索(BM25)"],
                key="retrieval_mode"
            )
        with col2:
            st.slider(
                "向量权重",
                0.0, 1.0, DEFAULT_VECTOR_WEIGHT,
                key="vector_weight"
            )

    # 文件上传
    with st.expander("📤 上传文档", expanded=False):
        uploaded_file = st.file_uploader(
            "选择PDF文件",
            type=["pdf"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            with st.spinner("正在处理文档..."):
                process_pdf_file(uploaded_file)

    # 问答区域
    st.divider()
    question = st.text_input(
        "💡 请输入您的问题：",
        placeholder="输入问题后按回车查询",
        key="question_input"
    )

    if question:
        # 检查是否需要重新生成回答
        if ("last_question" not in st.session_state or
                st.session_state.last_question != question or
                "last_answer" not in st.session_state):

            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    # 获取上下文
                    retriever = get_retriever()
                    docs = retriever.invoke(question)

                    # 生成回答 - 这里直接传递问题字符串，而不是字典
                    answer = rag_chain.invoke(question)

                    # 存储结果
                    st.session_state.last_question = question
                    st.session_state.last_answer = answer
                    st.session_state.last_context = docs
                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")
                    return

        # 显示回答
        st.markdown("### 回答")
        st.write(st.session_state.last_answer)

        # 显示上下文（不需要重新计算）
        if st.checkbox("显示相关上下文", key="show_context"):
            st.markdown("### 相关上下文")
            for i, doc in enumerate(st.session_state.last_context, 1):
                st.markdown(f"#### 上下文 {i}")
                st.markdown(f"**来源**: `{doc.metadata.get('source', '未知')}`")
                st.markdown(f"**页码**: {doc.metadata.get('page', '未知')}")
                st.text(doc.page_content)


# ==================== 应用入口 ====================
if __name__ == "__main__":
    # 确保session state已初始化
    initialize_session_state()

    # 加载界面
    knowledge_base_sidebar()
    main_interface()