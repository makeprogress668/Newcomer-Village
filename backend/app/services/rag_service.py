"""Agentic RAG 服务 (旅行知识库)。

为什么不用 ChromaDB / FAISS / Pinecone?
本场景知识库规模 ≤100 段落,纯关键词 + 简易 TF-IDF 已经能拿到很好效果,
省掉向量数据库的部署成本和依赖体积。

数据源: backend/app/data/knowledge_base/*.md
- 每个 .md 文件按 ## 二级标题切段
- 段落带"城市标签 + 主题标签"

检索 API:
  rag.search(query, city, top_k=3) -> [Document]
  rag.format_context(docs) -> str  (拼成 LLM context)

使用场景:
  - PlannerAgent 生成 description / overall_suggestions 时检索
  - GuardrailAgent 给"季节穿衣""安全提示"等增强建议
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KBDocument:
    """单条知识段落。"""
    city: str           # "beijing" / "shanghai" / "common"
    title: str          # 二级标题: "升旗仪式" / "故宫" 等
    content: str        # markdown 段落正文
    keywords: List[str] = None  # 切词后的关键词

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = _tokenize(self.title + " " + self.content)


# ============ 简易中文分词 (基于规则) ============

_TOKEN_PATTERN = re.compile(r"[一-龥]+|[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """简易切词: 中文按 2-gram, 英文按词。"""
    tokens: List[str] = []
    for piece in _TOKEN_PATTERN.findall(text or ""):
        if re.match(r"^[a-zA-Z0-9]+$", piece):
            tokens.append(piece.lower())
        else:
            # 中文 2-gram
            if len(piece) <= 2:
                tokens.append(piece)
            else:
                for i in range(len(piece) - 1):
                    tokens.append(piece[i:i + 2])
    return tokens


# ============ 知识库加载 ============

def _load_kb(kb_dir: Path) -> List[KBDocument]:
    docs: List[KBDocument] = []
    if not kb_dir.exists():
        logger.warning("知识库目录不存在: %s", kb_dir)
        return docs

    for md_file in sorted(kb_dir.glob("*.md")):
        city = md_file.stem  # 文件名去后缀: "beijing" / "common"
        text = md_file.read_text(encoding="utf-8")
        # 按 ## 切段
        sections = re.split(r"\n##\s+", text)
        # 第一段是 # 一级标题,跳过
        for section in sections[1:]:
            lines = section.strip().split("\n", 1)
            title = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            if title and content:
                docs.append(KBDocument(city=city, title=title, content=content))
    logger.info("📚 RAG 知识库加载完成: %d 段 (来自 %d 个文件)",
                len(docs), len(list(kb_dir.glob('*.md'))))
    return docs


# ============ 检索引擎 (TF-IDF + 城市过滤) ============

class TravelKnowledgeRAG:
    """旅行知识库检索器。"""

    def __init__(self, kb_dir: Optional[Path] = None):
        if kb_dir is None:
            kb_dir = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
        self.docs: List[KBDocument] = _load_kb(kb_dir)
        # 预计算 IDF
        self._idf: Dict[str, float] = self._compute_idf()

    def _compute_idf(self) -> Dict[str, float]:
        if not self.docs:
            return {}
        N = len(self.docs)
        df: Counter = Counter()
        for doc in self.docs:
            for token in set(doc.keywords):
                df[token] += 1
        return {tok: math.log(N / (1 + n)) + 1 for tok, n in df.items()}

    def search(
        self,
        query: str,
        city: Optional[str] = None,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> List[KBDocument]:
        """
        检索相关知识。
        Args:
            query: 用户查询文本
            city: 优先匹配该城市的知识 (如 "北京" → 文件名 "beijing")
            top_k: 返回前 K 条
            score_threshold: 最低相关度,过低的丢弃
        """
        if not self.docs or not query:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        city_tag = self._city_to_tag(city) if city else None

        scored: List = []
        for doc in self.docs:
            # 城市相关性: 匹配城市 +5 分,通用文档 +1 分
            city_boost = 0.0
            if city_tag and doc.city == city_tag:
                city_boost = 5.0
            elif doc.city == "common":
                city_boost = 1.0

            # TF-IDF 相似度
            doc_tf = Counter(doc.keywords)
            score = 0.0
            for token in query_tokens:
                if token in doc_tf:
                    tf = doc_tf[token]
                    idf = self._idf.get(token, 1.0)
                    score += tf * idf

            score += city_boost
            if score >= score_threshold:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def format_context(self, docs: List[KBDocument], max_chars: int = 1500) -> str:
        """把检索结果拼成可塞进 LLM prompt 的上下文。"""
        if not docs:
            return ""
        parts: List[str] = []
        used = 0
        for doc in docs:
            section = f"### {doc.title}\n{doc.content}\n"
            if used + len(section) > max_chars:
                break
            parts.append(section)
            used += len(section)
        return "\n".join(parts)

    @staticmethod
    def _city_to_tag(city: str) -> str:
        """中文城市名 → 文件名 tag (北京 → beijing)。"""
        mapping = {
            "北京": "beijing", "上海": "shanghai", "广州": "guangzhou",
            "深圳": "shenzhen", "杭州": "hangzhou", "成都": "chengdu",
            "西安": "xian", "重庆": "chongqing",
        }
        return mapping.get(city, "common")


# ============ 单例 ============

_rag: Optional[TravelKnowledgeRAG] = None


def get_rag() -> TravelKnowledgeRAG:
    global _rag
    if _rag is None:
        _rag = TravelKnowledgeRAG()
    return _rag
