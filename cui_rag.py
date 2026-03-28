from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import os
import pickle

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

def extract_text_with_page_numbers(pdf, pdf_name: str = "") -> Tuple[str, List[dict]]:
    """
    从PDF中提取文本并记录每个字符对应的页码和文件名
    
    参数:
        pdf: PDF文件对象
        pdf_name: PDF文件名
    
    返回:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码和文件信息列表
    """
    text = ""
    char_page_mapping = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            # 为当前页面的每个字符记录页码和文件名
            char_page_mapping.extend([{"file": pdf_name, "page": page_number}] * len(extracted_text))
        else:
            print(f"文件 {pdf_name} 第 {page_number} 页未找到文本。")

    return text, char_page_mapping

def process_text_with_splitter(text: str, char_page_mapping: List[dict], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码和文件信息列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 为每个文本块找到对应的页码和文件信息
    page_info = {}
    current_pos = 0
    
    for chunk in chunks:
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)
        
        # 找到这个文本块中字符对应的页码和文件信息
        chunk_pages = char_page_mapping[chunk_start:chunk_end]
        
        # 统计每个文件-页码组合出现的次数
        if chunk_pages:
            page_counts = {}
            for page_info_item in chunk_pages:
                key = (page_info_item["file"], page_info_item["page"])
                page_counts[key] = page_counts.get(key, 0) + 1
            
            # 找到出现次数最多的文件-页码组合
            most_common = max(page_counts, key=page_counts.get)
            page_info[chunk] = {"file": most_common[0], "page": most_common[1]}
        else:
            page_info[chunk] = {"file": "未知", "page": 1}  # 默认值
        
        current_pos = chunk_end
    
    knowledgeBase.page_info = page_info
    print(f'页码映射完成，共 {len(page_info)} 个文本块')
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")
    
    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
    
    return knowledgeBase

# 定义知识库文档目录和保存路径
knowledge_base_dir = "./知识库文档"
save_dir = "./cui_vector_db"

# 定义要处理的PDF文件列表（关于崔世亚的文件）
pdf_files = [
    "奖励证明.pdf",
    "崔世亚_实习生录用通知函.pdf",
    "扫描件_研究生学位论文开题报告.pdf",
    "本科阶段崔世亚个人简历.pdf",
    "研究生_实习证明.pdf",
    "研究生论文-基于LSTM-transformer模型的物流无人机能耗预测.pdf",
    "研究生论文-基于强化学习协同进化算法的低空载人调度方法.pdf"
]

# 检查向量数据库是否已存在
if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "index.faiss")):
    print("检测到已存在的向量数据库，正在加载...")
    knowledgeBase = load_knowledge_base(save_dir)
    print("向量数据库加载完成！")
else:
    print("未检测到向量数据库，正在创建...")
    
    # 合并所有PDF文件的文本和页码映射
    all_text = ""
    all_char_page_mapping = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(knowledge_base_dir, pdf_file)
        if os.path.exists(pdf_path):
            print(f"正在处理: {pdf_file}")
            try:
                pdf_reader = PdfReader(pdf_path)
                text, char_page_mapping = extract_text_with_page_numbers(pdf_reader, pdf_file)
                all_text += text
                all_char_page_mapping.extend(char_page_mapping)
                print(f"  - 提取了 {len(text)} 个字符")
            except Exception as e:
                print(f"  - 处理失败: {str(e)}")
        else:
            print(f"文件不存在: {pdf_path}")
    
    print(f"\n总共提取的文本长度: {len(all_text)} 个字符。")
    
    # 处理文本并创建知识库，同时保存到磁盘
    knowledgeBase = process_text_with_splitter(all_text, all_char_page_mapping, save_path=save_dir)
    print("\n崔世亚个人信息知识库创建完成！")

# ============ 问答系统 ============


from langchain_community.llms import Tongyi
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY) # qwen-turbo

# 设置查询问题
query = "崔世亚的教育背景是什么？"
# query = "崔世亚有哪些研究成果？"
# query = "崔世亚的实习经历是什么？"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query,k=10)

    # 构建上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构建提示
    prompt = f"""根据以下上下文回答问题:

{context}

问题: {query}"""

    # 直接调用 LLM
    response = llm.invoke(prompt)
    print(response)
    print("来源:")

    # 记录唯一的页码
    unique_pages = set()

    # 显示每个文档块的来源文件和页码
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_info = knowledgeBase.page_info.get(
            text_content.strip(), {"file": "未知", "page": "未知"}
        )
        
        source_key = (source_info.get("file", "未知"), source_info.get("page", "未知"))
        if source_key not in unique_pages:
            unique_pages.add(source_key)
            print(f"  - 文件: {source_info.get('file', '未知')}, 页码: {source_info.get('page', '未知')}")

