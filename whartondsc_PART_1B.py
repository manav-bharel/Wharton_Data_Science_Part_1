# @title I hope this is the final code 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Load data
past_games_df = pd.read_csv('games_2022.csv')
upcoming_games_df = pd.read_csv('East Regional Games to predict.csv')

# Define relevant statistical columns
stats_columns = [
    'FGA_2', 'FGA_3', 'FTA', 'FTM', 'FGM_2', 'FGM_3', 'AST', 'STL', 'TOV',
    'DREB', 'OREB', 'BLK', 'F_personal', 'F_tech', 'team_score', 'opponent_team_score'
]

# Average stats per team from past games
team_avg_stats = past_games_df.groupby('team')[stats_columns].mean().reset_index()

# Create eFG%, TOV rate, and FT/FGA
past_games_df['eFG%'] = ((past_games_df['FGM_2'] + past_games_df['FGM_3']) + (0.5*past_games_df['FGM_3']))/ (past_games_df['FGA_2'] + past_games_df['FGA_3'])
past_games_df['possessions'] = (past_games_df['FGA_2'] + past_games_df['FGA_3'] + past_games_df['FTA'] * 0.44 + past_games_df['TOV'] - past_games_df['OREB'])/2
past_games_df['TOV_ratio'] = past_games_df['TOV'] / past_games_df['possessions']
past_games_df['FT/FGA'] = past_games_df['FTM'] / (past_games_df['FGA_2'] + past_games_df['FGA_3'])

# Add new stats to average stats
stats_columns = stats_columns + ['eFG%', 'TOV_ratio', 'FT/FGA']

team_avg_stats = past_games_df.groupby('team')[stats_columns].mean().reset_index()

# Merge average team stats into upcoming games
for prefix, team_col in [('home', 'team_home'), ('away', 'team_away')]:
    rename_dict = {col: f"{col}_{prefix}_avg" for col in stats_columns}
    upcoming_games_df = upcoming_games_df.merge(
        team_avg_stats.rename(columns=rename_dict),
        left_on=team_col, right_on='team', how='left'
    ).drop(columns=['team'])

# Split past games into home and away teams
past_games_home = past_games_df[past_games_df['home_away'] == 'home'].copy()
past_games_away = past_games_df[past_games_df['home_away'] == 'away'].copy()

# Merge home and away teams based on game_id
past_games_merged = past_games_home.merge(
    past_games_away, on='game_id', suffixes=('_home', '_away')
)

# Create ORB% and add average to upcoming games
past_games_merged['ORB%_home'] = (past_games_merged['OREB_home'])/(past_games_merged['OREB_home'] + past_games_merged['OREB_away'])
past_games_merged['ORB%_away'] = (past_games_merged['OREB_away'])/(past_games_merged['OREB_home'] + past_games_merged['OREB_away'])

avg_orb_home = past_games_merged.groupby('team_home', as_index=False)['ORB%_home'].mean()
avg_orb_away = past_games_merged.groupby('team_away', as_index=False)['ORB%_away'].mean()

upcoming_games_df = upcoming_games_df.merge(
    avg_orb_home.rename(columns={'ORB%_home': 'ORB%_home'}),
    left_on='team_home', right_on='team_home', how='left'
)

upcoming_games_df = upcoming_games_df.merge(
    avg_orb_away.rename(columns={'ORB%_away': 'ORB%_away'}),
    left_on='team_away', right_on='team_away', how='left'
)


# Rename game_date column properly
past_games_merged = past_games_merged.rename(columns={'game_date_home': 'game_date'})

# Create win/loss labels
past_games_merged['home_win'] = (past_games_merged['team_score_home'] > past_games_merged['team_score_away']).astype(int)

# New variables for past games
past_games_merged['rest_days_diff'] = past_games_merged['rest_days_home'] - past_games_merged['rest_days_away']
past_games_merged['travel_dist_diff'] = past_games_merged['travel_dist_away'] - past_games_merged['travel_dist_home']

games_per_team = past_games_merged.groupby("team_home")["game_id"].count()
min_games = 10  
valid_teams = games_per_team[games_per_team >= min_games].index
past_games_merged = past_games_merged[past_games_merged["team_home"].isin(valid_teams) & 
                                      past_games_merged["team_away"].isin(valid_teams)]


# Set test/train data based on game date
past_games_merged['game_date'] = pd.to_datetime(past_games_merged['game_date'])
past_games_merged = past_games_merged.sort_values(by='game_date')
cutoff_date = past_games_merged['game_date'].quantile(0.8)
train_data = past_games_merged[past_games_merged['game_date'] <= cutoff_date]
test_data = past_games_merged[past_games_merged['game_date'] > cutoff_date]

# Features for training the model (using past game stats)
features = [
    'rest_days_diff', 'travel_dist_diff', 'ORB%_home', 'ORB%_away'
] + [f"{col}_{prefix}" for col in stats_columns for prefix in ['home', 'away']]

# Define train/test
X_train = train_data[features]
y_train = train_data['home_win']

X_test = test_data[features]
y_test = test_data['home_win']

# Standardize features
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)


X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

# Build NN
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='leaky_relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='leaky_relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='leaky_relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-4, clipvalue=1.0)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# New variables for upcoming games
upcoming_games_df['rest_days_diff'] = upcoming_games_df['rest_days_Home'] - upcoming_games_df['rest_days_Away']
upcoming_games_df['travel_dist_diff'] = upcoming_games_df['travel_dist_Away'] - upcoming_games_df['travel_dist_Home']

# Rename columns in upcoming_games_df to match the training feature names
rename_dict = {f"{col}_home_avg": f"{col}_home" for col in stats_columns}
rename_dict.update({f"{col}_away_avg": f"{col}_away" for col in stats_columns})

upcoming_games_df = upcoming_games_df.rename(columns=rename_dict)

# Features for upcoming games
features_upcoming = [
    'rest_days_diff', 'travel_dist_diff', 'ORB%_home', 'ORB%_away'
] + [f"{col}_{prefix}" for col in stats_columns for prefix in ['home', 'away']]

X_upcoming_scaled = scaler.transform(upcoming_games_df[features_upcoming])

raw_predictions = model.predict(X_upcoming_scaled)

prob_predictions = tf.nn.sigmoid(raw_predictions).numpy()

# Predict win rate
upcoming_games_df['home_win_chance'] = prob_predictions

# Save predictions
upcoming_games_df[['team_home', 'team_away', 'home_win_chance']].to_csv('predictions_for_all_upcoming_games.csv', index=False)

# Print predictions
print(upcoming_games_df[['team_home', 'team_away', 'home_win_chance']])
