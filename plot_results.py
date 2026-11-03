"""Create report figures from experiment CSV outputs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(folder):
    folder=Path(folder)
    comparison=pd.read_csv(folder/'model_comparison.csv')
    predictions=pd.read_csv(folder/'holdout_predictions.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(comparison['method'],comparison['RMSE'])
    ax.set(ylabel='Holdout RMSE',title='Feature Selection Comparison')
    ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(folder/'rmse_comparison.png',dpi=160);plt.close(fig)
    best=comparison.iloc[0]['method']
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(predictions['actual'],predictions[best],s=6,alpha=0.25)
    bounds=[min(predictions['actual'].min(),predictions[best].min()),max(predictions['actual'].max(),predictions[best].max())]
    ax.plot(bounds,bounds,linestyle='--',color='black')
    ax.set(xlabel='Actual',ylabel='Predicted',title=f'Actual vs Predicted — {best}')
    fig.tight_layout();fig.savefig(folder/'actual_vs_predicted.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(predictions[best],predictions['actual']-predictions[best],s=6,alpha=0.25)
    ax.axhline(0,linestyle='--',color='black')
    ax.set(xlabel='Predicted',ylabel='Residual',title=f'Residuals — {best}')
    fig.tight_layout();fig.savefig(folder/'residuals.png',dpi=160);plt.close(fig)
    intervals=folder/'prediction_intervals.csv'
    if intervals.exists():
        d=pd.read_csv(intervals).sort_values('predicted').reset_index(drop=True)
        x=np.arange(len(d))
        fig,ax=plt.subplots(figsize=(11,5))
        ax.fill_between(x,d['prediction_lower'],d['prediction_upper'],alpha=0.2,label='95% OLS prediction interval')
        ax.plot(x,d['predicted'],label='OLS predicted',linewidth=1)
        ax.scatter(x,d['actual'],s=5,label='Actual')
        ax.set(xlabel='Holdout observations sorted by predicted value',ylabel='Target',title='Conditional 95% Prediction Intervals')
        ax.legend();fig.tight_layout();fig.savefig(folder/'prediction_intervals.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='outputs')
    plot(parser.parse_args().output_dir)
