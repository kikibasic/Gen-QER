import os
import logging
from src import utils, benchmark
from src.retriever import NeuralRetriever
from src.prompts import PromptManager
from src.generator import LLMGenerator
from src.searcher import SparseSearcher
from src.evaluation import Evaluator
import config


logging.getLogger('httpx').setLevel(logging.WARNING) 
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',force=True)

def main(args):
    # 1. モデル初期化
    logging.info(f"Initializing Retriever: {args.rank_model} (Mode: {args.mode})")
    retriever = NeuralRetriever(model_name=args.rank_model, mode=args.mode)
    generator = None
    if args.doc_gen > 0:
        try:
            generator = LLMGenerator(args.llm)
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            return

    # データセット
    data_list = ['dl20', 'dl19', 'covid', 'nfc' ,'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04'] #data_list = ['dl20', 'dl19', 'covid', 'nfc' ,'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']

    for dataset in data_list:
        logging.info(f"#" * 30)
        logging.info(f"Processing Dataset: {dataset}")
        logging.info(f"#" * 30)
        
        # 2. Sparse Retrieval & Psued Reference Generation
        try:
            bm25_results = SparseSearcher.get_results_with_generation(
                dataset=dataset, generator=generator, prompt_manager=PromptManager, args=args
            )
        except Exception as e:
            logging.error(f"Error in Sparse Search for {dataset}: {e}")
            continue

        if not bm25_results:
            logging.warning(f"No results found for {dataset}. Skipping.")
            continue

        # 3. Reranking
        if args.irmode in ['mugirerank', 'mugipipeline']:
            logging.info(f"Starting Dense Reranking... (Top-K: {args.dense_topk})")
            
            # 生成文のキー名 (ex: gen_cand_gpt4)
            gen_key = f'gen_cand_{args.llm}' if 'gpt' not in args.llm else 'gen_cand_gpt4'
            
            # Rerank実行
            rerank_result = retriever.rerank(
                bm25_results, 
                gen_key, 
                topk=args.dense_topk, 
                use_enhanced_query=True
            )

            # 4. JSONの保存 (結果 + 疑似参照文)
            rerank_json = utils.normalize_rerank_to_bm25_json(rerank_result, bm25_results)
            
            # 保存
            run_tag = f"{dataset}_{args.llm}_{args.mode}_n{args.doc_gen}"
            json_path = os.path.join(args.output_path, args.llm, f"{run_tag}.json")
            
            logging.info(f"💾 Saving JSON to: {json_path}")
            utils.dump_json(rerank_json, json_path)

            # 5. TREC RUN形式への変換
            run_dir = os.path.join("results", "runs", args.llm)
            run_path = os.path.join(run_dir, f"{run_tag}.run")
            
            logging.info(f"🔄 Converting to RUN format: {run_path}")
            utils.convert_json_to_run(json_path, run_path, dataset)

            # 6. 評価
            logging.info("📊 Evaluating...")
            qrels_path = Evaluator.get_qrels_path(dataset)
            
            if qrels_path and os.path.exists(run_path):
                score = Evaluator.run_trec_eval(run_path, qrels_path)
                logging.info(f"✅ {dataset} Result | nDCG@10: {score:.4f}")
                
                # ログを保存
                summary_filename = f"{args.irmode}.json"
                summary_path = os.path.join("results", summary_filename)
                summary_data = utils.load_json(summary_path)

                if args.llm not in summary_data: 
                    summary_data[args.llm] = {}
                if args.rank_model not in summary_data[args.llm]: 
                    summary_data[args.llm][args.rank_model] = {}
                key_mode = f"{args.mode}_n{args.doc_gen}"

                if key_mode not in summary_data[args.llm][args.rank_model]: 
                    summary_data[args.llm][args.rank_model][key_mode] = {}
                
                summary_data[args.llm][args.rank_model][key_mode][dataset] = score
                
                # 保存
                utils.dump_json(summary_data, summary_path)
                logging.info(f"📊 Updated summary: {summary_path}")
            else:
                logging.warning(f"Skipping evaluation for {dataset} (Qrels or Run file missing).")

        logging.info(f"Finished {dataset}.\n")

if __name__ == "__main__":
    args = config.parse_args()
    main(args)