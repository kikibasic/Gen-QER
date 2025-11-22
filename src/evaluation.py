import os
import re
import subprocess
import platform
import logging
from pyserini.search import get_qrels_file
from pyserini.util import download_evaluation_script
from src import benchmark

class Evaluator:
    @staticmethod
    def get_qrels_path(dataset_name):
        """
        データセット名から自動でqrelsファイルのパスを取得する。
        ダウンロードされていない場合はPyseriniが自動ダウンロードする。
        """
        topic_key = benchmark.THE_TOPICS.get(dataset_name)
        if not topic_key:
            logging.error(f"No topic definition found for {dataset_name}")
            return None
        
        try:
            # Pyseriniがキャッシュからパスを返す（なければダウンロード）
            return get_qrels_file(topic_key)
        except Exception as e:
            logging.error(f"Failed to get qrels for {dataset_name}: {e}")
            return None

    @staticmethod
    def run_trec_eval(run_path, qrels_path, metric='ndcg_cut.10'):
        """
        指定されたRUNファイルとQRELSファイルを使ってtrec_evalを実行する。
        """
        if not os.path.exists(run_path) or not os.path.exists(qrels_path):
            logging.error("Run file or Qrels file missing.")
            return 0.0

        script_path = download_evaluation_script('trec_eval')
        cmd = ['java', '-Dfile.encoding=UTF-8', '-jar', script_path, '-c', '-m', metric, qrels_path, run_path]
        
        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        stdout, stderr = process.communicate()
        
        if stderr:
            # Javaのエラーなどはログに出す
            logging.debug(stderr.decode("utf-8"))
            
        output = stdout.decode("utf-8").rstrip()
        # 出力からスコア数値を抽出 (例: "ndcg_cut_10  all  0.7123" -> 0.7123)
        match = re.search(r'\d+\.\d+', output.split('\t')[-1]) if '\t' in output else re.search(r'\d+\.\d+', output)
        
        score = float(match.group(0)) if match else 0.0
        return score