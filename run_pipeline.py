"""End-to-end fraud pipeline from source dataset to documented test plots."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'src'))
from data_audit import audit
from experiment import run
from plot_results import plot
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/raw/creditcard.csv');p.add_argument('--out',default='results');p.add_argument('--fast',action='store_true');a=p.parse_args()
    if not Path(a.csv).exists():raise SystemExit('Dataset missing. Run: python src/download_data.py')
    df=pd.read_csv(a.csv); audit(df,a.out)
    r=run(df,a.out,fast=a.fast); plot(Path(a.out)/'comparison.csv',a.out)
    print(json.dumps(r,indent=2))
