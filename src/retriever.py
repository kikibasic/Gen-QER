import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List, Dict

TASK_DESC = 'Given a web search query, retrieve relevant passages that answer the query'

def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    return torch.sum(last_hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class NeuralRetriever:
    def __init__(self, model_name, mode):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode

    def embed(self, input_texts):
        # 元のreranker.pyのロジックをそのまま利用
        input_tokens = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**input_tokens)
            if "bge" in self.model_name.lower():
                embeddings = outputs[0][:, 0]
            else:
                embeddings = mean_pooling(outputs.last_hidden_state, input_tokens['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _enhance_query_text(self, q, refs):
        """単純連結用のヘルパー"""
        if self.mode == "concat":
            return q + " ".join(refs) 
        elif self.mode == "qg":
            return q + (refs[0] if refs else "")
        return q

    def rerank(self, rank_result: List[Dict], gen_key: str, topk=100, use_enhanced_query=False):
        rerank_result = {}
        
        for item in tqdm(rank_result, desc="Reranking"):
            q = item.get("query", "")
            
            # Context-Pool Implementation
            if use_enhanced_query and self.mode == 'contex-pool':
                refs = item.get(gen_key) or []
                # クエリと各参照文のペアを作成
                enhanced_queries = [q + " " + r for r in refs]
                if not enhanced_queries: enhanced_queries = [q]
                
                # バッチエンコードして平均化
                all_embeddings = self.embed(enhanced_queries)
                query_embed = torch.mean(all_embeddings, dim=0, keepdim=True)
                query_embed = F.normalize(query_embed, p=2, dim=1)
            
            # 既存のモード
            else:
                if use_enhanced_query:
                    refs = item.get(gen_key) or []
                    query_text = self._enhance_query_text(q, refs)
                else:
                    query_text = q
                query_embed = self.embed(query_text)

            # ドキュメントのエンコードとスコアリング
            current_hits = item['hits'][:topk]
            if not current_hits: continue
            
            docs = [hit['content'] for hit in current_hits]
            docs_idx = [hit['docid'] for hit in current_hits]
            
            hits_embed = self.embed(docs)
            scores = torch.matmul(query_embed, hits_embed.T)
            
            # Top-10 取得
            _, indices = scores.topk(min(10, len(docs)), dim=1)
            selected_doc_ids = [docs_idx[i] for i in indices.reshape(-1).tolist()]
            
            # QIDをキーに保存
            qid = item['hits'][0]['qid'] if item['hits'] else item.get('qid')
            if qid: rerank_result[qid] = selected_doc_ids
            
        return rerank_result