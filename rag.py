from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch


load_dotenv()

ZILLIZ_ENDPOINT = os.getenv('ZILLIZ_ENDPOINT')
ZILLIZ_USER = os.getenv('ZILLIZ_USER')
ZILLIZ_PASS = os.getenv('ZILLIZ_PASS')

# # load pdf file
# file = "xxx.pdf"
# loader = PDFPlumberLoader(file)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(docs)

# def get_huggingface_embedding(text, model_id="Snowflake/snowflake-arctic-embed-m-v2.0"): 
#     """
#     使用 Hugging Face transformers 加载 snowflake-arctic-embed2 模型并获取文本 embedding
#     :param text: 要生成 embedding 的文本
#     :param model_id: Hugging Face Model ID of snowflake-arctic-embed2 (replace placeholder)
#     :return: 文本的 embedding 向量 (list of float) 或 None (如果出错)
#     """
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModel.from_pretrained(model_id)

#         # Tokenize the text
#         inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

#         # Get model outputs
#         with torch.no_grad(): # Disable gradient calculation for inference
#             outputs = model(**inputs)

#         # For sentence embeddings, you typically want the embeddings of the [CLS] token
#         # or perform some pooling over the token embeddings.
#         # The exact method might depend on the specific model and documentation.
#         # Here, we'll assume taking the embedding of the [CLS] token (index 0)
#         embeddings = outputs.last_hidden_state[:, 0, :].tolist() # Get embedding of [CLS] token, convert to list

#         return embeddings[0] # Return the embedding vector for the first (and only) input text
#     except Exception as e:
#         print(f"Hugging Face embedding generation failed: {e}")
#         return None

# if __name__ == "__main__":
#     sample_text = "这是一个用于测试 Hugging Face embedding 模型的例子。"
#     model_id_to_use = "Snowflake/snowflake-arctic-embed-m-v2.0" # 
#     embedding_vector = get_huggingface_embedding(sample_text, model_id=model_id_to_use)

#     if embedding_vector:
#         print(f"文本: '{sample_text}'")
#         print(f"Embedding 向量 (前 10 个维度):")
#         print(embedding_vector[:10])
#         print(f"Embedding 向量维度: {len(embedding_vector)}")
#     else:
#         print("生成 Hugging Face embedding 向量失败。")



def get_huggingface_embedding(text, model_id="Snowflake/snowflake-arctic-embed-m-v2.0"): # Use the correct Model ID
    """
    使用 Hugging Face transformers 加载 snowflake-arctic-embed-m-v2.0 模型并获取文本 embedding
    :param text: 要生成 embedding 的文本
    :param model_id: Hugging Face Model ID of snowflake-arctic-embed-m-v2.0
    :return: 文本的 embedding 向量 (list of float) 或 None (如果出错)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True) # ADD trust_remote_code=True
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)     # ADD trust_remote_code=True

        # Tokenize the text
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        # Get model outputs
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(**inputs)

        # For sentence embeddings, you typically want the embeddings of the [CLS] token
        # or perform some pooling over the token embeddings.
        # The exact method might depend on the specific model and documentation.
        # Here, we'll assume taking the embedding of the [CLS] token (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :].tolist() # Get embedding of [CLS] token, convert to list

        return embeddings[0] # Return the embedding vector for the first (and only) input text
    except Exception as e:
        print(f"Hugging Face embedding generation failed: {e}")
        return None

if __name__ == "__main__":
    sample_text = "这是一个用于测试 Hugging Face embedding 模型的例子。"
    model_id_to_use = "Snowflake/snowflake-arctic-embed-m-v2.0" # Correct Model ID from previous analysis
    embedding_vector = get_huggingface_embedding(sample_text, model_id=model_id_to_use)

    if embedding_vector:
        print(f"文本: '{sample_text}'")
        print(f"Embedding 向量 (前 10 个维度):")
        print(embedding_vector[:10])
        print(f"Embedding 向量维度: {len(embedding_vector)}")
    else:
        print("生成 Hugging Face embedding 向量失败。")

