"""
Human Behavioral Data Loading
==============================
Loads and returns structured data from the Hu et al. experiment.
"""
import os
import pandas as pd

_HUMAN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'human-data', 'temporal-bias-human-analysis',
    'data', 'data', 'behavioral_18sub')

_MU_XLSX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'human-data', 'temporal-bias-human-analysis',
    'data', 'processed_memdata', 'memresults.xlsx')

POSITIONS = [0.20, 0.40, 0.60, 0.80]


def load_human_signed_error(data_dir=_HUMAN_DATA_DIR):
    if not os.path.exists(data_dir):
        return None
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
    dfs = []
    for f in files:
        pid = f.split('_')[0]
        df = pd.read_excel(os.path.join(data_dir, f), header=0)
        df['subject'] = pid
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    subj_means = all_data.groupby(['subject', 'td'])['ERROR'].mean().reset_index()
    means = subj_means.groupby('td')['ERROR'].mean()
    sems = subj_means.groupby('td')['ERROR'].sem()
    return {td: {'mean': means[td], 'sem': sems[td]} for td in POSITIONS}


def load_human_mu(mu_path=_MU_XLSX_PATH):
    if not os.path.exists(mu_path):
        return None
    mu_df = pd.read_excel(mu_path, sheet_name='Sheet1')
    means = mu_df.groupby('td')['mu'].mean()
    sems = mu_df.groupby('td')['mu'].sem()
    return {td: {'mean': means[td], 'sem': sems[td]} for td in POSITIONS}
