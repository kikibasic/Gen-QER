import os
import json
import logging
from typing import List, Dict, Any
from pyserini.search import get_topics
from src import benchmark

def load_json(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def dump_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def convert_json_to_run(json_path: str, run_path: str, dataset: str):
    """
    JSON結果ファイルをTREC RUN形式に変換して保存する。
    """
    if not os.path.exists(os.path.dirname(run_path)):
        os.makedirs(os.path.dirname(run_path), exist_ok=True)

    # Dataset名からTopic名を取得 (benchmark.pyを利用)
    topic_key = benchmark.THE_TOPICS.get(dataset)
    if not topic_key:
        logging.warning(f"Unknown dataset: {dataset}. Conversion might fail if queries don't match.")
        return

    # クエリ文字列 -> QID のマッピングを作成
    topics = get_topics(topic_key)
    q2id = {}
    for qid, obj in topics.items():
        qtext = obj.get("title") or obj.get("query") or str(obj)
        q2id[qtext.strip()] = str(qid)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    # runファイルのタグ名（ファイル名から拡張子を除いたもの）
    tag = os.path.splitext(os.path.basename(run_path))[0]

    for entry in data:
        qtext = entry.get('query', '').strip()
        # QIDの特定: 結果JSONにqidが含まれていればそれを優先、なければテキストマッチ
        qid = str(entry.get('hits', [{}])[0].get('qid', ''))
        if not qid:
            qid = q2id.get(qtext)
        
        if not qid:
            continue

        for rank, hit in enumerate(entry.get('hits', []), start=1):
            docid = hit.get('docid')
            score = hit.get('score', 0.0)
            if docid:
                records.append(f"{qid} Q0 {docid} {rank} {score} {tag}\n")

    with open(run_path, "w", encoding="utf-8") as f:
        f.writelines(records)
    
    logging.info(f"Converted to RUN: {run_path} ({len(records)} lines)")

# ... (以下、前回の normalize_rerank_to_bm25_json 等はそのまま残す) ...
def _index_bm25_by_qid(bm25_rank_results: Any) -> Dict[str, Dict[str, Any]]:
    # ... (前回のコードと同じ) ...
    qmap: Dict[str, Dict[str, Any]] = {}
    results = bm25_rank_results if isinstance(bm25_rank_results, list) else [bm25_rank_results]
    for entry in results:
        hits = entry.get('hits', [])
        if not hits: continue
        qid = str(hits[0].get('qid'))
        if not qid: continue
        hits_by_docid = {str(h.get('docid')): h for h in hits}
        extra = {k: v for k, v in entry.items() if k not in ('hits',)}
        qmap[qid] = {'query': entry.get('query', ''), 'extra': extra, 'hits_by_docid': hits_by_docid}
    return qmap

def normalize_rerank_to_bm25_json(rerank_result: Any, bm25_rank_results: Any) -> List[Dict[str, Any]]:
    # ... (前回のコードと同じ) ...
    out: List[Dict[str, Any]] = []
    qmap = _index_bm25_by_qid(bm25_rank_results)
    if isinstance(rerank_result, dict):
        for qid, docids in rerank_result.items():
            qid = str(qid)
            base = qmap.get(qid, {'query': '', 'extra': {}, 'hits_by_docid': {}})
            entry: Dict[str, Any] = {'query': base['query'], 'hits': []}
            entry.update(base['extra']) 
            for rnk, docid in enumerate(docids, start=1):
                src = base['hits_by_docid'].get(str(docid), {})
                entry['hits'].append({
                    'qid': qid, 'docid': docid, 'rank': rnk,
                    'score': src.get('score', 0.0), 'content': src.get('content', '')
                })
            out.append(entry)
    return out