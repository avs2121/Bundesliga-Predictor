import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


matches = pd.read_csv(r'data_matches\de.1_total.csv')

print(matches.head())

print(matches.columns)

matches = matches.drop('status',axis=1); # 1 Abonned match -> correct result registered.
matches['date'] = pd.to_datetime(matches['date'])

matches.rename(columns={'team1': 'home_team', 'team2': 'away_team'},inplace=True)

matches_home = matches.copy() 
matches_home['team'] = matches_home['home_team']
matches_home['opponent'] = matches_home['away_team']
matches_home['is_home'] = 1
matches_home['goals_for'] = matches_home['score/ft/0']
matches_home['goals_against'] = matches_home['score/ft/1']

matches_away = matches.copy()
matches_away['team'] = matches_away['away_team']
matches_away['opponent'] = matches_away['home_team']
matches_away['is_home'] = 0
matches_away['goals_for'] = matches_away['score/ft/1']
matches_away['goals_against'] = matches_away['score/ft/0']

matches_final = pd.concat([matches_home, matches_away],ignore_index=True)


def get_team_result(row):
    if(row['goals_for'] > row['goals_against']):
        return 2 #2 For Win
    elif(row['goals_for'] < row['goals_against']):
        return 0 #0 For Loss
    elif(row['goals_for'] == row['goals_against']):
        return 1 #1 for Draw

matches_final['team_result'] = matches_final.apply(get_team_result,axis=1)
#matches_final['team_result'] = matches_final['team_result'].map({'W':2, 'D':1, 'L':0})



# Encode team names into integers
le = LabelEncoder()
matches_final["opponent"] = le.fit_transform(matches_final["opponent"])
matches_final["team"] = le.fit_transform(matches_final["team"]) 

model = XGBClassifier(n_estimators=4000, learning_rate=0.01, max_depth=4, subsample=0.8, 
                    min_child_weight=2, colsample_bytree=0.8, random_state=0, reg_alpha=0.5, reg_lambda=1.5)

train = matches_final[matches_final['date'] < '2024-07-01']
test = matches_final[matches_final['date'] > '2024-07-01']

predictors = ['is_home', 'team' , 'opponent']

model.fit(train[predictors], train['team_result'])

preds = model.predict(test[predictors])

acc = accuracy_score(test["team_result"], preds) ## testing accuracy

print(acc)


