import argparse
import os
import sys
import csv
import math
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import Evaluator
from src.utils import convert_json_to_run
from src import benchmark

def load_qrels(path):
    q = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4: continue
            qid, docid, rel = p[0], p[2], int(p[3])
            q.setdefault(qid, {})[docid] = rel
    return q

def load_run(path):
    r = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6: continue
            qid, docid, rank, score = p[0], p[2], int(p[3]), float(p[4])
            r[qid].append((docid, score, rank))
    out = {}
    for qid, items in r.items():
        items.sort(key=lambda x: (-x[1], x[2]))
        out[qid] = [d for d, _, _ in items]
    return out

def dcg_at_k(gains, k):
    s = 0.0
    for i, g in enumerate(gains[:k], start=1):
        if g > 0:
            s += g / math.log2(i + 1)
    return s

def ndcg_at_k_from_qrels(qrels_for_q, ranked_docs, k):
    gains = [qrels_for_q.get(d, 0) for d in ranked_docs]
    dcg = dcg_at_k(gains, k)
    ideal = sorted([rel for rel in qrels_for_q.values() if rel > 0], reverse=True)
    if len(ideal) < k:
        ideal += [0] * (k - len(ideal))
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

def condensed_ndcg(qrels_for_q, ranked_docs, k):
    judged_docs = [d for d in ranked_docs if d in qrels_for_q]
    if not judged_docs:
        return 0.0
    return ndcg_at_k_from_qrels(qrels_for_q, judged_docs, k)

def eval_run(qrels, run, k):
    rows = []
    for qid, docs in run.items():
        qr = qrels.get(qid, {})
        ndcg = ndcg_at_k_from_qrels(qr, docs, k)
        unjudged_k = sum(1 for d in docs[:k] if d not in qr)
        first_hit = ""
        for rank, d in enumerate(docs, start=1):
            if qr.get(d, 0) > 0:
                first_hit = rank
                break
        hit1 = 1 if (len(docs) >= 1 and qr.get(docs[0], 0) > 0) else 0
        judged_at_k = k - unjudged_k
        cndcg = condensed_ndcg(qr, docs, k)
        # catastrophic zero: 全くかすりもしない場合（未判定が多い場合を考慮）
        cat_zero = 1 if (ndcg == 0.0 and unjudged_k >= 5) else 0
        
        rows.append({
            "qid": qid,
            f"ndcg@{k}": f"{ndcg:.4f}",
            f"condensed_ndcg@{k}": f"{cndcg:.4f}",
            f"unjudged@{k}": unjudged_k,
            f"judged@{k}": judged_at_k,
            "first_hit_rank": first_hit,
            "hit@1": hit1,
            "catastrophic_zero": cat_zero
        })
    return rows

def to_float(x):
    try: return float(x)
    except: return 0.0

def summarize(rows, k, include_delta=False):
    n = len(rows)
    avg_ndcg = sum(to_float(r[f"ndcg@{k}"]) for r in rows) / n if n else 0.0
    avg_cndcg = sum(to_float(r[f"condensed_ndcg@{k}"]) for r in rows) / n if n else 0.0
    avg_unjudged = sum(r[f"unjudged@{k}"] for r in rows) / n if n else 0.0
    avg_first_hit_vals = [r["first_hit_rank"] for r in rows if r["first_hit_rank"] != ""]
    avg_first_hit = (sum(avg_first_hit_vals) / len(avg_first_hit_vals)) if avg_first_hit_vals else 0.0
    
    print("\n=== 📊 SUMMARY ===")
    print(f"Dataset Queries\t{n}")
    print(f"Avg nDCG@{k}\t{avg_ndcg:.4f}")
    print(f"Avg Condensed nDCG@{k}\t{avg_cndcg:.4f}")
    print(f"Avg Unjudged@{k}\t{avg_unjudged:.2f}")
    print(f"Avg First Hit Rank\t{avg_first_hit:.2f}")
    
    if include_delta:
        deltas = [to_float(r[f"delta_ndcg@{k}"]) for r in rows if f"delta_ndcg@{k}" in r]
        if deltas:
            mean_delta = sum(deltas)/len(deltas)
            print(f"Mean Delta nDCG@{k}\t{mean_delta:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Gen-QER Results (Detailed)")
    parser.add_argument('--json', type=str, required=True, help='Path to result JSON file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (dl19, dl20, etc.)')
    parser.add_argument('--output_dir', type=str, default='results/runs', help='Directory to save analysis files')
    parser.add_argument('--k', type=int, default=10, help='Cutoff k for metrics')
    parser.add_argument('--baseline_run', default=None, help='Path to baseline run file for comparison')
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"❌ Error: File not found: {args.json}")
        return

    # 1. JSON -> RUN 変換
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(args.json).replace('.json', '.run')
    run_path = os.path.join(args.output_dir, filename)
    
    print(f"🔄 Converting JSON to TREC Run format...")
    convert_json_to_run(args.json, run_path, args.dataset)
    print(f"   Saved: {run_path}")

    # 2. Qrelsの自動取得
    print(f"🔍 Fetching qrels for {args.dataset}...")
    qrels_path = Evaluator.get_qrels_path(args.dataset)
    
    if not qrels_path or not os.path.exists(qrels_path):
        print(f"❌ Error: Qrels not found for {args.dataset}")
        return

    # 3. 詳細評価の実行
    qrels = load_qrels(qrels_path)
    run_data = load_run(run_path)
    rows = eval_run(qrels, run_data, args.k)

    # ベースライン比較
    if args.baseline_run and os.path.exists(args.baseline_run):
        print(f"📉 Comparing with baseline: {args.baseline_run}")
        base_run = load_run(args.baseline_run)
        base_rows = eval_run(qrels, base_run, args.k)
        base_map = {r["qid"]: r for r in base_rows}
        for r in rows:
            qid = r["qid"]
            if qid in base_map:
                nd = to_float(r[f"ndcg@{args.k}"])
                bd = to_float(base_map[qid][f"ndcg@{args.k}"])
                r[f"delta_ndcg@{args.k}"] = f"{(nd - bd):.4f}"

    # 4. CSV保存
    csv_filename = filename.replace('.run', '_metrics.csv')
    csv_path = os.path.join(args.output_dir, csv_filename)
    
    fields = ["qid", f"ndcg@{args.k}", f"condensed_ndcg@{args.k}", f"unjudged@{args.k}", f"judged@{args.k}", "first_hit_rank", "hit@1", "catastrophic_zero"]
    if args.baseline_run:
        fields.append(f"delta_ndcg@{args.k}")
        
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
    print(f"💾 Saved per-query metrics to: {csv_path}")

    # 5. サマリー表示
    summarize(rows, args.k, include_delta=bool(args.baseline_run))

if __name__ == "__main__":
    main()