import os
import glob
import hashlib
import shutil
from typing import Dict, List, Optional
import chromadb
from tqdm import tqdm
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BGE_M3_Processor:
    """使用BAAI/bge-m3模型处理PDF并构建向量数据库的处理器"""
    
    # 模型配置
    MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024  # bge-m3的嵌入维度
    
    def __init__(self, 
                 pdf_folder: str = "./pdf_files",
                 db_path: str = "chroma_db_bge_m3",
                 processed_dir: str = "processed_pdfs"):
        """
        初始化PDF处理器
        
        :param pdf_folder: 存放待处理PDF的目录
        :param db_path: ChromaDB数据库路径
        :param processed_dir: 处理后的PDF存储目录
        """
        self.pdf_folder = pdf_folder
        self.db_path = db_path
        self.processed_dir = processed_dir
        
        # 初始化嵌入函数
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.MODEL_NAME,
            device="cuda" if self._check_gpu() else "cpu"
        )
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 确保目录存在
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(pdf_folder, exist_ok=True)

    def _check_gpu(self) -> bool:
        """检查GPU是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def process_all_pdfs(self, 
                       chunk_size: int = 800,
                       chunk_overlap: int = 100,
                       collection_name: str = "bge_m3_docs") -> Dict:
        """
        处理目录中的所有PDF文件
        
        :param chunk_size: 文本分块大小
        :param chunk_overlap: 分块重叠大小
        :param collection_name: Chroma集合名称
        
        :return: 处理结果字典，包含:
            - total: 总文件数
            - processed: 成功处理数
            - skipped: 跳过数(重复文件)
            - failed: 失败数
            - details: 每个文件的处理详情
            - collection_info: 集合信息
        """
        pdf_files = glob.glob(os.path.join(self.pdf_folder, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"未找到PDF文件于: {self.pdf_folder}")

        # 创建/获取集合
        collection = self._get_or_create_collection(
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        results = {
            "total": len(pdf_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "details": []
        }

        for pdf_file in tqdm(pdf_files, desc="处理PDF文件"):
            file_result = self._process_single_file(
                pdf_file=pdf_file,
                collection=collection,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 更新统计结果
            results[file_result["status"]] += 1
            results["details"].append(file_result)

        # 添加集合信息
        results["collection_info"] = {
            "name": collection_name,
            "count": collection.count(),
            "dimension": self.EMBEDDING_DIM,
            "model": self.MODEL_NAME
        }
        
        return results

    def _get_or_create_collection(self, 
                                collection_name: str,
                                chunk_size: int,
                                chunk_overlap: int) -> chromadb.Collection:
        """获取或创建Chroma集合"""
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "dimensionality": str(self.EMBEDDING_DIM),
                "embedding_model": self.MODEL_NAME,
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap)
            }
        )

    def _process_single_file(self,
                           pdf_file: str,
                           collection: chromadb.Collection,
                           chunk_size: int,
                           chunk_overlap: int) -> Dict:
        """处理单个PDF文件"""
        file_result = {
            "file": os.path.basename(pdf_file),
            "status": None,
            "message": "",
            "chunks": 0
        }
        
        try:
            # 检查是否已处理过相同内容文件
            file_hash = self._get_file_hash(pdf_file)
            existing = collection.get(
                where={"content_hash": {"$eq": file_hash}},
                include=["metadatas"]
            )
            
            if existing["ids"]:
                file_result.update({
                    "status": "skipped",
                    "message": f"相同内容文件已存在: {existing['metadatas'][0][0]['source_file']}"
                })
                return file_result

            # 处理PDF文件
            splits = self._split_pdf(
                pdf_path=pdf_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 存储到向量数据库
            collection.add(
                documents=[doc.page_content for doc in splits],
                metadatas=[{
                    "page": doc.metadata["page"],
                    "source_file": os.path.basename(pdf_file),
                    "content_hash": file_hash
                } for doc in splits],
                ids=[f"{file_hash}_{i}" for i in range(len(splits))]
            )

            # 移动已处理文件
            self._move_processed_file(pdf_file, file_hash)
            
            file_result.update({
                "status": "processed",
                "message": f"存储块数: {len(splits)}",
                "chunks": len(splits)
            })
            
        except Exception as e:
            file_result.update({
                "status": "failed",
                "message": str(e)
            })
        
        return file_result

    def _split_pdf(self, 
                  pdf_path: str, 
                  chunk_size: int, 
                  chunk_overlap: int) -> List:
        """分割PDF文档为文本块"""
        loader = PyPDFLoader(pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        return splitter.split_documents(loader.load())

    def _move_processed_file(self, file_path: str, file_hash: str):
        """移动已处理文件到指定目录"""
        new_path = os.path.join(
            self.processed_dir,
            f"{file_hash}_{os.path.basename(file_path)}"
        )
        shutil.move(file_path, new_path)

    def query(self,
             query_text: str,
             collection_name: str = "bge_m3_docs",
             n_results: int = 3,
             filter_by_file: Optional[str] = None) -> Dict:
        """
        查询向量数据库
        
        :param query_text: 查询文本
        :param collection_name: 集合名称
        :param n_results: 返回结果数量
        :param filter_by_file: 按文件名过滤
        
        :return: 查询结果字典，包含:
            - documents: 匹配文档列表
            - metadatas: 元数据列表
            - distances: 距离分数列表
            - ids: 文档ID列表
        """
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        where = {"source_file": {"$eq": filter_by_file}} if filter_by_file else None
        
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

    def get_collection_info(self, collection_name: str = "bge_m3_docs") -> Dict:
        """获取集合信息"""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "dimension": self.EMBEDDING_DIM,
                "model": self.MODEL_NAME
            }
        except Exception as e:
            return {"error": str(e)}

    def reset_database(self):
        """重置整个数据库"""
        self.client.reset()
        shutil.rmtree(self.pdf_folder, ignore_errors=True)
        shutil.rmtree(self.processed_dir, ignore_errors=True)
        os.makedirs(self.pdf_folder, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
