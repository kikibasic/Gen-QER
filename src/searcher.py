import logging
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from src import benchmark
from src.prompts import PromptManager
from src.utils import dump_json, load_json
import os

class SparseSearcher:
    
    @staticmethod
    def get_data_pyserini(data, test=False):
        searcher = LuceneSearcher.from_prebuilt_index(benchmark.THE_INDEX[data])
        topics = get_topics(benchmark.THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(benchmark.THE_TOPICS[data])
        topics = {k: v for k, v in topics.items() if k in qrels}
        if test:
            topics = {key: topics[key] for key in list(topics)[:10]}
        return searcher, topics, qrels

    @staticmethod
    def get_results_with_generation(dataset, generator, prompt_manager, args):
        """Executes generation (if needed) and sparse retrieval."""
        
        # 1. Load Data
        searcher, topics, qrels = SparseSearcher.get_data_pyserini(dataset, args.test)
        
        # 2. Generate Pseudo References
        gen_key = None
        if generator:
            gen_key = f'gen_cand_{args.llm}' if 'gpt' not in args.llm else 'gen_cand_gpt4'
            logging.info(f"Generating pseudo-docs for {dataset} using {args.llm}...")
            
            for key in tqdm(topics, desc="Generating"):
                # クエリの取得
                query = topics[key]['title']
                # プロンプトの取得
                messages = prompt_manager.get_prompt(query)
                
                topics[key].setdefault(gen_key, [])
                # 指定回数生成
                current_count = len(topics[key][gen_key])
                if current_count < args.doc_gen:
                    for _ in range(args.doc_gen - current_count):
                        output = generator.generate(messages).strip()
                        topics[key][gen_key].append(output)
        
        # 3. Run BM25
        return SparseSearcher.bm25_search(args, topics, searcher, qrels, gen_key)

    @staticmethod
    def bm25_search(args, topics, searcher, qrels, gen_key=None):
        """Runs BM25. Expands query if gen_key is provided."""
        for key in topics:
            query = topics[key]['title']
            
            # クエリ拡張
            if gen_key and gen_key in topics[key]:
                gen_refs = topics[key][gen_key][:args.article_num]
                gen_text = ' '.join(gen_refs)
                
                # Adaptive or Fixed repetition
                if args.repeat_times:
                    times = args.repeat_times
                elif args.adaptive_times:
                    times = max(1, (len(gen_text)//max(1, len(query)))//max(1, args.adaptive_times))
                else:
                    times = 1
                topics[key]['enhanced_query'] = (query + ' ')*times + gen_text
            else:
                # 拡張なし
                topics[key]['enhanced_query'] = query

        # 検索実行
        logging.info(f"Running BM25 search...")
        rank_results = SparseSearcher._run_pyserini_search(topics, searcher, gen_key, args.topk, use_enhanced_query=(gen_key is not None))
        return rank_results

    @staticmethod
    def _run_pyserini_search(topics, searcher, gen_key, k=100, use_enhanced_query=False):
        ranks = []
        for qid, topic in tqdm(topics.items(), desc="BM25 Search"):
            query_text = topic['enhanced_query'] if use_enhanced_query else topic['title']
            
            try:
                hits = searcher.search(query_text, k=k)
            except Exception as e:
                logging.error(f"Search failed for qid {qid}: {e}")
                hits = []

            rank_details = {'query': topic['title'], 'qid': qid, 'hits': []}
            
            # 生成テキストを結果JSONに保持（後続のRerankerで使うため）
            if gen_key and gen_key in topic:
                rank_details[gen_key] = topic[gen_key]

            for rank, hit in enumerate(hits, start=1):
                # Raw doc content extraction
                raw = searcher.doc(hit.docid).raw()
                try:
                    if isinstance(raw, bytes): raw = raw.decode('utf-8')
                    content = json.loads(raw)
                    text_content = content.get('text', content.get('contents', ''))
                    if 'title' in content:
                        text_content = f"Title: {content['title']} Content: {text_content}"
                except:
                    text_content = "" # Fallback
                
                text_content = ' '.join(text_content.split())
                
                rank_details['hits'].append({
                    'content': text_content,
                    'qid': qid,
                    'docid': hit.docid,
                    'rank': rank,
                    'score': hit.score
                })
            ranks.append(rank_details)
        return ranks