import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Gen-QER: Query Expansion & Reranking")

    parser.add_argument('--irmode', type=str, default='mugipipeline',
                        choices=['mugisparse','rerank','mugirerank','mugipipeline'],
                        help='Information retrieval mode')
    
    # Document Generation Settings
    parser.add_argument('--llm', type=str, default='gpt-4o', help='Pseudo reference generation model (gpt, 01-ai, Qwen, etc.)')
    parser.add_argument('--doc_gen', type=int, default=2, help='Number of generated documents (n)')
    parser.add_argument('--output_path', type=str, default='./exp', help='Output directory path')
    
    # Sparse Retrieval (BM25) Settings
    parser.add_argument('--repeat_times', '-t', default=None, type=int, help='Fixed repetition times for query expansion')
    parser.add_argument('--adaptive_times', '-at', default=6, type=int, help='Adaptive repetition factor')
    parser.add_argument('--topk', type=int, default=100, help='BM25 retrieved top-k documents')
    parser.add_argument('--article_num','-a', default=5, type=int, help='Number of pseudo-docs used for sparse expansion')
    
    # Dense Retrieval / Reranking Settings
    parser.add_argument('--rank_model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='HuggingFace model name for dense retrieval/reranking')
    parser.add_argument('--dense_topk', type=int, default=100, help='Number of documents to rerank')
    
    # ★独自性: contex-pool を追加
    parser.add_argument('--mode', type=str, 
                        choices=['query', 'alternate', 'concat', 'qg', 'contex-pool'], 
                        default='contex-pool',
                        help='Query enhancement mode. Use "contex-pool" for best performance.')
    
    parser.add_argument('--test', action='store_true', help='Run in fast test mode (fewer queries)')
    
    return parser.parse_args()