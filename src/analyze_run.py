import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import Evaluator
from scripts.convert_run import convert, TOPIC_NAMES

def main():
    parser = argparse.ArgumentParser(description="Analyze Gen-QER Results JSON")
    parser.add_argument('--json', type=str, required=True, help='Path to the result JSON file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (dl19, dl20, etc.)')
    parser.add_argument('--output_dir', type=str, default='results/runs', help='Directory to save .trecrun files')
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: File not found: {args.json}")
        return

    # 1. Runファイルへの変換
    filename = os.path.basename(args.json).replace('.json', '.trecrun')
    run_path = os.path.join(args.output_dir, filename)
    
    print(f"Converting JSON to TREC Run format...")
    print(f"  Input:  {args.json}")
    print(f"  Output: {run_path}")
    
    convert(args.json, run_path, args.dataset)

    # 2. 評価の実行 (ndcg@10)
    if args.dataset in TOPIC_NAMES:
        topic_file = TOPIC_NAMES[args.dataset]
        print(f"Running trec_eval on {args.dataset}...")
        
        # Evaluatorを使って評価実行
        # args: [-c, -m, ndcg_cut.10, qrels_file, run_file]
        # qrelsはEvaluator内部で自動取得されるが、ここでは明示的にTopic名を渡す必要がある
        # Evaluator.run_trec_eval は qrelsパス解決ロジックを持つ
        
        # Note: Evaluator.evaluate_dictはdict入力用なので、
        # ここではファイルベースの評価コマンドを直接呼ぶか、Evaluatorを拡張して使う
        # 簡易的に trec_eval コマンドを構築して実行
        
        score = Evaluator.run_trec_eval(['-c', '-m', 'ndcg_cut.10', topic_file, run_path])
        print(f"\n>>> nDCG@10 Score: {score:.4f}")
    else:
        print(f"Warning: Dataset {args.dataset} not found in known topics. Skipping evaluation.")

if __name__ == "__main__":
    main()