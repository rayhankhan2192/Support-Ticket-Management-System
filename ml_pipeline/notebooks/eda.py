from __future__ import annotations

import json


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Save plots without needing a graphical desktop
import seaborn as sns
import re
from pathlib import Path
from scipy.stats import entropy

import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

REQUIRED = ['Ticket category', 'Example message', 'Assigned team']
FILLER_WORDS = ['really', 'just', 'always', 'still', 'suddenly', 'currently', 'extremely']

def save_chart(fig, chart_path: Path, filename: str) -> None:
    fig.savefig(chart_path / filename, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)

def tokens_of(s: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", s.lower())


def normalized_text(s: str) -> str:
    return ' '.join(tokens_of(s))


def fmt_pct(n: int, d: int) -> str:
    return f'{n / d * 100:.1f}%' if d else 'n/a'


# Load, validate and derive NLP features from the raw data

def load_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    df = pd.read_csv(path, encoding='utf-8-sig')
    if not all(col in df.columns for col in REQUIRED):
        raise ValueError(f"Data file must contain the following columns: {REQUIRED}")
    df = df[REQUIRED].copy()
    missing = df.isnull().sum().rename('missing_count').to_frame()
    missing['missing_pct'] = missing['missing_count'] / len(df) * 100
    logger.info(f"Missing values:\n{missing}")
    for col in REQUIRED:
        df[col] = df[col].fillna('').astype(str).str.strip()
    df = df.rename(columns = {
        'Ticket category': 'category',
        'Example message': 'message',
        'Assigned team': 'team'
    })
    df['normalized_message'] = df.message.map(normalized_text)
    df['tokens'] = df.message.map(tokens_of)
    df['char_count'] = df.message.str.len()
    df['word_count'] = df.tokens.map(len)
    df['unique_word_count'] = df.tokens.map(lambda xs: len(set(xs)))
    df['lexical_diversity'] = np.where(df.word_count > 0, df.unique_word_count / df.word_count.clip(lower=1), 0.0)
    df['mean_word_length'] = df.tokens.map(lambda xs: float(np.mean([len(w) for w in xs])) if xs else 0.0)
    df['question_mark'] = df.message.str.contains(r'\?', regex=True)
    df['exclamation_mark'] = df.message.str.contains('!', regex=False)
    df['number_present'] = df.message.str.contains(r'\d', regex=True)
    df['has_email'] = df.message.str.contains(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', regex=True)
    df['has_url'] = df.message.str.contains(r'https?://|www\.', regex=True, case=False)
    df['starts_with'] = df.tokens.map(lambda x: ' '.join(x[:2]))
    for word in FILLER_WORDS:
        df[f'has_{word}'] = df.message.str.contains(r'\b' + re.escape(word) + r'\b', case=False, regex=True)
    
    return df, missing


# Data auditing and descriptive statistics 
def audit_data(df: pd.DataFrame, missing: pd.DataFrame, data_dir: Path) -> dict:
    counts = df.category.value_counts()
    n = len(df)
    duplicates = df.message.duplicated().sum()
    duplicate_normalized = df.normalized_message.duplicated().sum()
    blank_messages = (df.message.str.strip() == '').sum()
    mapping = pd.crosstab(df.category, df.team, margins=True)
    mapping.to_csv(data_dir / 'category_team_mapping.csv', index=True)
    missing.to_csv(data_dir / 'missing_values.csv', index=True)
    df[df.message.str.strip() == ''].to_csv(data_dir / 'blank_messages.csv', index=False)
    df[df.normalized_message.duplicated(keep=False)].drop(columns=['tokens']).to_csv(
        data_dir / 'duplicate_normalized_messages.csv', index=False)
    counts.rename_axis('category').reset_index(name='tickets').to_csv(data_dir / 'category_counts.csv', index=False)
    fields = ['char_count', 'word_count', 'unique_word_count', 'lexical_diversity', 'mean_word_length']
    stats = df.groupby('category')[fields].agg(['mean', 'median', 'std', 'min', 'max'])
    stats.to_csv(data_dir / 'text_statistics_by_category.csv')
    df[fields].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_csv(data_dir / 'overall_text_statistics.csv')
    df.drop(columns=['tokens']).to_csv(data_dir / 'engineered_ticket_features.csv', index=False)
    results = {
        'dataset_rows': int(n), 'category_count': int(counts.size),
        'team_count': int(df.team.nunique()),
        'raw_missing_values': {str(k): int(v) for k, v in missing.missing_count.items()},
        'blank_messages': blank_messages,
        'duplicate_exact_messages': duplicates,
        'duplicate_normalized_messages': duplicate_normalized,
        'category_counts': {str(k): int(v) for k,v in counts.items()},
        'mean_words_per_ticket': round(float(df.word_count.mean()), 2),
        'median_words_per_ticket': float(df.word_count.median()),
        'mean_characters_per_ticket': round(float(df.char_count.mean()), 2),
        'class_max_min_ratio': round(float(counts.max() / counts.min()), 3),
        'normalized_category_entropy': round(float(entropy(counts.values) / np.log(max(2, len(counts)))), 4),
        'categories_with_multiple_teams': int((mapping.gt(0).sum(axis=1)>1).sum()),
        'teams_with_multiple_categories': int((mapping.gt(0).sum(axis=0)>1).sum()),
        'has_email_count': int(df.has_email.sum()),
        'has_url_count': int(df.has_url.sum()),
    }
    (data_dir / 'data_audit.json').write_text(json.dumps(results, indent=4), encoding='utf-8')
    return results