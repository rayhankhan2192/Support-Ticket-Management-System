# %%
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
from sklearn.feature_extraction.text import CountVectorizer

logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

REQUIRED = ['Ticket category', 'Example message', 'Assigned team']
FILLER_WORDS = ['really', 'just', 'always', 'still', 'suddenly', 'currently', 'extremely']
SEED = 42
PALETTE = sns.color_palette('colorblind', 8)

# %%
def save_chart(fig, chart_path: Path, filename: str) -> None:
    fig.savefig(chart_path / filename, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)

def tokens_of(s: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", s.lower())


def normalized_text(s: str) -> str:
    return ' '.join(tokens_of(s))


def fmt_pct(n: int, d: int) -> str:
    return f'{n / d * 100:.1f}%' if d else 'n/a'


# %% Load, validate and derive NLP features from the raw data
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


# %% Data auditing and descriptive statistics 
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
    (data_dir / 'summary_metrics.json').write_text(json.dumps(results, indent=4), encoding='utf-8')
    return results

# %%
def plot_category_distribution(df: pd.DataFrame, chart_path: Path) -> None:
    order = df.category.value_counts().index.to_list()
    counts = df.category.value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(11, 5.3))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='colorblind', legend=False)
    # Swap x and y to make the bars horizontal
    #sns.barplot(y=counts.index, x=counts.values, ax=ax, palette='colorblind', legend=False)
    for p, value in zip(ax.patches, counts.values):
        ax.text(
            p.get_width()+10, 
            p.get_y()+p.get_height()/2, 
            f'{value:,} ({value/len(df):.1%})', 
            va='center', 
            fontsize=9)
        # ax.text(
        #     p.get_x() + p.get_width() / 2, 
        #     p.get_height() + 10, # Note: if your counts are small, 10 might be too high
        #     f'{value:,} ({value/len(df):.1%})', 
        #     ha='center', 
        #     va='bottom', 
        #     fontsize=9
        # )
    ax.set_xlim(0, counts.max() * 1.20)  # Add some space to the right for the text
    ax.set(title='01. Ticket category distribution', xlabel='Ticket count', ylabel='')
    save_chart(fig, chart_path, '01_category_distribution.png')
    
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('colorblind'))
    ax.set_title('02. Category proportion')
    save_chart(fig, chart_path, '02_category_donut.png')
    
    mapping = pd.crosstab(df.category, df.team)
    fig, ax = plt.subplots(figsize=(11,6))
    sns.heatmap(mapping, annot=True, fmt='d', cmap='Blues', linewidths=.6, ax=ax, cbar=False)
    ax.set(title='03. Category–team mapping (do not use team to predict category)', xlabel='Assigned team', ylabel='Ticket category')
    ax.tick_params(axis='x', labelrotation=30)
    save_chart(fig, chart_path, '03_category_team_heatmap.png')

    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(df.char_count, bins=35, kde=True, color=PALETTE[0], ax=ax)
    ax.axvline(df.char_count.median(), ls='--', lw=1.6, label=f'Median: {df.char_count.median():.0f} chars')
    ax.set(title='04. Message length distribution', xlabel='Characters per ticket')
    ax.legend()
    save_chart(fig, chart_path, '04_character_lengths.png')

    fig, ax = plt.subplots(figsize=(11,5.5))
    sns.violinplot(data=df, y='category', x='word_count', order=order,
                    hue='category', palette='colorblind', legend=False, inner='quart', cut=0, ax=ax)
    ax.set(title='05. Word count by ticket category', xlabel='Words per message', ylabel='')
    save_chart(fig, chart_path, '05_word_count_by_category.png')

    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(data=df, x='word_count', binwidth=1, discrete=True, color=PALETTE[0], ax=ax)
    ax.set(title='06. Overall words per ticket', xlabel='Word count', ylabel='Tickets')
    save_chart(fig, chart_path, '06_word_count_distribution.png')

    fig, ax = plt.subplots(figsize=(10,5.5))
    sample = df.sample(min(len(df), 2500), random_state=SEED)
    sns.scatterplot(data=sample, x='word_count', y='char_count', hue='category',
                    alpha=.43, s=21, palette='colorblind', ax=ax)
    ax.set(title='07. Word count vs character count (sample)', xlabel='Words', ylabel='Characters')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), fontsize=8, title='Ticket category')
    save_chart(fig, chart_path, '07_word_vs_character_scatter.png')

    fig, ax = plt.subplots(figsize=(11,5.5))
    sns.boxplot(data=df, x='lexical_diversity', y='category', order=order,
                hue='category', palette='colorblind', legend=False, ax=ax)
    ax.set(title='08. Message lexical diversity (unique words / total words)',
            xlabel='Lexical diversity per message', ylabel='')
    save_chart(fig, chart_path, '08_lexical_diversity.png')

    usage = df.groupby('category')[['question_mark','exclamation_mark','number_present']].mean().mul(100)
    fig, ax = plt.subplots(figsize=(9,5.2))
    sns.heatmap(usage, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, cbar_kws={'label':'% of tickets'})
    ax.set(title='09. Punctuation and numerical information across categories', xlabel='Text feature', ylabel='')
    save_chart(fig, chart_path, '09_text_feature_heatmap.png')


# %% Frequency analysis of words and phrases
def freq_table( texts: pd.Series, ngram: tuple[int, int], stopwords: str = 'english', top_n: int = 25) -> pd.DataFrame:
    try:
        vectorizer = CountVectorizer(ngram_range=ngram, stop_words=stopwords)
        X = vectorizer.fit_transform(texts)
    except ValueError as e:
        logger.error(f"Error in CountVectorizer: {e}")
        return pd.DataFrame(columns=['ngram', 'count'])
    
    freqs = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(freqs)[::-1][:top_n]
    return pd.DataFrame({
        'ngram': np.array(vectorizer.get_feature_names_out())[idx],
        'count': freqs[idx].astype(int),
    })
    
    
def plot_frequencies(df: pd.DataFrame, chart_path: Path, table_path: Path) -> None:
    for ngram, label, num in [((1,1), 'Unigrams', '10'), ((2,2), 'Bigrams', '11')]:
        word_counts = freq_table(df.message, ngram, top_n=25)
        word_counts.to_csv(table_path / f'top_{label.lower()}.csv', index=False)
        fig, ax = plt.subplots(figsize=(11,5.5))
        if not word_counts.empty:
            sns.barplot(data=word_counts.head(20).iloc[::-1], x='count', y='term', color=PALETTE[0], ax=ax)
        ax.set(title=f'{num}. Top 20 {label.lower()} (English stopwords removed)', xlabel='Corpus frequency', ylabel='')
        save_chart(fig, chart_path, f'{num}_top_{label.lower()}.png')
        
        all_counts = []
        for category in sorted(df.category.unique()):
            freqs = freq_table(df.loc[df.category.eq(category), 'message'], ngram, top_n=20)
            freqs.insert(0, 'category', category)
            all_counts.append(freqs)
        
        if not all_counts:
            logger.warning(f"No frequency data found for {label.lower()}. Skipping plot.")
            return
        per_category_counts = pd.concat(all_counts, ignore_index=True)
        per_category_counts.to_csv(table_path / f'top_{label.lower()}_by_category.csv', index=False)
        groups = sorted(df.category.unique())
        fig, ax = plt.subplots(figsize=(11,5.5))
        for category, ax in zip(groups, ax.ravel()):
            subset = per_category_counts[per_category_counts.category.eq(category)]
            if not subset.empty:
                sns.barplot(data=subset.head(20).iloc[::-1], x='count', y='term', color=PALETTE[0], ax=ax)
            ax.set(title=f'{num}. Top 20 {label.lower()} for category: {category}', xlabel='Corpus frequency', ylabel='')
        fig.suptitle('12. Top unigrams for each category',y=1.002,fontsize=15)
        fig.tight_layout()
        save_chart(fig, chart_dir, '12_category_top_unigrams.png')