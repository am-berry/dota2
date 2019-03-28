import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import eli5
import catboost as cb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
r= 17
N_ESTIMATORS = 1500

train_df = pd.read_csv('train_features.csv.zip', index_col='match_id_hash')
y_train = pd.read_csv('train_targets.csv', index_col='match_id_hash')['radiant_win']
y_train = y_train.map({True: 1, False: 0})
test_df = pd.read_csv('test_features.csv.zip', index_col = 'match_id_hash')

def team_totals(df):
    df['radiant_lh'] = df['r1_lh'] + df['r2_lh'] + df['r3_lh'] + df['r4_lh'] + df['r5_lh']
    df['dire_lh'] = df['d1_lh'] + df['d2_lh'] + df['d3_lh'] + df['d4_lh'] + df['d5_lh']
    df['radiant_lvl'] = df['r1_level'] + df['r2_level'] + df['r3_level'] + df['r4_level'] + df['r5_level']
    df['dire_lvl'] = df['d1_level'] + df['d2_level'] + df['d3_level'] + df['d4_level'] + df['d5_level']
    df['radiant_roshans'] = df['r1_roshans_killed'] + df['r2_roshans_killed'] + df['r3_roshans_killed'] + df['r4_roshans_killed'] + df['r5_roshans_killed']
    df['dire_roshans'] = df['d1_roshans_killed'] + df['d2_roshans_killed'] + df['d3_roshans_killed'] + df['d4_roshans_killed'] + df['d5_roshans_killed']
    df['radiant_gold'] = df['r1_gold'] + df['r2_gold'] + df['r3_gold'] + df['r4_gold'] + df['r5_gold']
    df['dire_gold'] = df['d1_gold'] + df['d2_gold'] + df['d3_gold'] + df['d4_gold'] + df['d5_gold']
    df['radiant_kills'] = df['r1_kills'] + df['r2_kills'] + df['r3_kills'] + df['r4_kills'] + df['r5_kills']
    df['dire_kills'] = df['d1_kills'] + df['d2_kills'] + df['d3_kills'] + df['d4_kills'] + df['d5_kills']
    df['radiant_deaths'] = df['r1_deaths'] + df['r2_deaths'] + df['r3_deaths'] + df['r4_deaths'] + df['r5_deaths']
    df['dire_deaths'] = df['d1_deaths'] + df['d2_deaths'] + df['d3_deaths'] + df['d4_deaths'] + df['d5_deaths']
    df['radiant_towers'] = df['r1_towers_killed'] + df['r2_towers_killed'] + df['r3_towers_killed'] + df['r4_towers_killed'] + df['r5_towers_killed']
    #df['radiant_objectives'] = df['radiant_roshans'] + df['radiant_towers']
    df['dire_towers'] = df['d1_towers_killed'] + df['d2_towers_killed'] + df['d3_towers_killed'] + df['d4_towers_killed'] + df['d5_towers_killed']
    #df['dire_objectives'] = df['objectives_len'] - df['radiant_objectives']
    #df['percent_radiant'] = df['radiant_objectives'] / df['objectives_len']
    #df['radiant_more_objectives'] = df['percent_radiant'].apply(lambda x: 1 if x >=0.5 else 0)
    df.drop(['lobby_type'], axis = 1, inplace=True)
    return df

def combine_numeric_features (df, feature_suffixes):
    for feat_suff in feature_suffixes:
        for team in 'r', 'd':
            players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...
            player_col_names = [f'{player}_{feat_suff}' for player in players] # e.g. r1_gold, r2_gold

            df[f'{team}_{feat_suff}_max'] = df[player_col_names].max(axis=1) # e.g. r_gold_max
            df[f'{team}_{feat_suff}_mean'] = df[player_col_names].mean(axis=1) # e.g. r_gold_mean
            df[f'{team}_{feat_suff}_min'] = df[player_col_names].min(axis=1) # e.g. r_gold_min
            df[f'{team}_{feat_suff}_total'] = df[player_col_names].sum(axis=1)
            df.drop(columns=player_col_names, inplace=True) # remove raw features from the dataset
    return df

def encode(df):
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        hero_columns = [f'{player}_hero_id' for player in players]
        d = pd.get_dummies(df[hero_columns[0]])
        for c in hero_columns[1:]:
            d += pd.get_dummies(df[c])
        df = pd.concat([df, d.add_prefix(f'{team}_hero_')], axis=1)
        df.drop(columns=hero_columns, inplace=True)
    return df

def gmencode(df):

    gmencoded = pd.get_dummies(df['game_mode'])
    df = pd.concat([df, gmencoded], axis=1)
    df.drop(['game_mode'], axis=1, inplace=True)
    df = df.rename(columns = {22: 'ap', 23: 'tm', 4: 'sd', 3: 'rd', 2: 'cm', 5:'ar', 12:'lp', 16:'cd'})
    return df

#train_df = team_totals(train_df)
train_df = combine_numeric_features(train_df, numeric_features)
train_df = encode(train_df)
train_df = gmencode(train_df)
#test_df = team_totals(test_df)
test_df = combine_numeric_features(test_df, numeric_features)
test_df = encode(test_df)
test_df = gmencode(test_df)

def evaluate():
    from catboost import CatBoostClassifier, Pool
    train_df_part, valid_df, y_train_part, y_valid = \
        train_test_split(train_df, y_train, test_size=0.25, random_state=r)
    cat_features_idx = np.where(train_df.dtypes == 'object')[0].tolist()
    catboost_dataset = Pool(train_df_part, label=y_train_part, cat_features=cat_features_idx)
    catboost_dataset_valid = Pool(valid_df, label=y_valid, cat_features=cat_features_idx)
    catboost_classifier = CatBoostClassifier(
        eval_metric='AUC', depth=5, learning_rate=0.02,
        random_seed=r, verbose=False, n_estimators=N_ESTIMATORS, task_type='GPU')
    catboost_classifier.fit(catboost_dataset, eval_set=catboost_dataset_valid, plot=True)
    valid_pred = catboost_classifier.predict_proba(valid_df)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)
    print('Score:', score)
    return catboost_classifier

classifier = evaluate()
print('Classifier training complete')
submission_df = pd.read_csv('sample_submission.csv', index_col='match_id_hash')
submission_df['radiant_win_prob'] = classifier.predict_proba(test_df)[:, 1]
print('Classifier prediction complete')
submission_df.to_csv('submission.csv')
print('Successfully output to csv file')
