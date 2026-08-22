import json
import os
import pandas as pd
import numpy as np

# Load dataset to get real top 20 for markdown review table
df = pd.read_csv('data/raw/content_refresh_anonymized.csv')

def percentile_rank(s):
    return s.rank(pct=True, na_option='keep')

def normalize(s):
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return np.zeros_like(s)
    return (s - s_min) / (s_max - s_min)

df['visibility_score'] = percentile_rank(np.log1p(df['impressions_90d'])).fillna(0)
df['freshness_risk_score'] = percentile_rank(df['days_since_last_update']).fillna(0)
df['position_opportunity_score'] = (
    (1 - normalize(df['avg_position'].clip(lower=1, upper=50)))
    * df['visibility_score']
    * (df['avg_position'] > 0).astype(int)
).fillna(0)

word_rank = percentile_rank(df['word_count'])
df['depth_gap_score'] = ((1 - word_rank) * df['visibility_score']).fillna(0)

df['baseline_refresh_score'] = (
    0.40 * df['visibility_score']
    + 0.30 * df['freshness_risk_score']
    + 0.25 * df['position_opportunity_score']
    + 0.05 * df['depth_gap_score']
).clip(0, 1)

def get_reason_codes(row):
    reasons = []
    if row['days_since_last_update'] >= 180 and row['impressions_90d'] >= 500:
        reasons.append('stale_visible_page')
    if row['avg_position'] > 0 and row['avg_position'] <= 10 and row['content_age_days'] >= 180:
        reasons.append('page_one_decay_risk')
    if row['impressions_90d'] >= 500 and 0 < row['avg_position'] <= 20 and row['ctr'] < 0.5:
        reasons.append('low_ctr_visible_page')
    if row['word_count'] > 0 and row['word_count'] < 1200 and row['impressions_90d'] >= 250:
        reasons.append('thin_visible_page')
    if row['sessions_90d'] >= 30 and ((0 < row['engagement_rate'] < 30) or (0 < row['scroll_rate'] < 30)):
        reasons.append('low_engagement_visible_page')
    if not reasons:
        reasons.append('general_refresh_review')
    return '|'.join(reasons)

def get_suggested_action(reasons_str):
    reasons = set(reasons_str.split('|'))
    if 'thin_visible_page' in reasons:
        return 'expand_and_refresh'
    if 'low_ctr_visible_page' in reasons:
        return 'refresh_and_review_ctr'
    if 'stale_visible_page' in reasons or 'page_one_decay_risk' in reasons:
        return 'refresh'
    return 'monitor'

df['reason_codes'] = df.apply(get_reason_codes, axis=1)
df['suggested_action_baseline'] = df['reason_codes'].apply(get_suggested_action)
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)
df['baseline_rank'] = df['baseline_refresh_score'].rank(method='first', ascending=False).astype(int)
df_sorted = df.sort_values('baseline_rank')

top20 = df_sorted.head(20)

print("Top 20 computed for template.")
