import numpy as np
import pandas as pd
from collections import defaultdict

num_states = 1000
num_actions = 9
episodes = 100000
discount_factor = 0.6
learning_rate = 0.01
epsilon = 1.0
show_episodes = 10

file_path = "games_2022 - games_2022.csv"
raw_data = pd.read_csv(file_path)

Q_table = defaultdict(lambda: np.zeros(num_actions))
weights = np.random.rand(3)

raw_data['game_id'] = raw_data['game_id'].astype(str).str.extract('(\d{4})$', expand=False)
raw_data['game_id'] = raw_data['game_id'].fillna(0).astype(int)

def find_teamB(random_game_index, raw_data):
    teamA = raw_data.iloc[random_game_index]
    teamA_id = teamA['game_id']

    next_column = raw_data.iloc[random_game_index + 1] if random_game_index + 1 < len(raw_data) else None
    if next_column is not None and next_column['game_id'] == teamA_id:
        return next_column

    previous_column = raw_data.iloc[random_game_index - 1] if random_game_index - 1 < len(raw_data) else None
    if previous_column is not None and previous_column['game_id'] == teamA_id:
        return previous_column

    return None

def extract_team_stats(team):
    return {
        "FGA_2": team["FGA_2"], "FGM_2": team["FGM_2"], "FGA_3": team["FGA_3"], "FGM_3": team["FGM_3"],
        "FTA": team["FTA"], "FTM": team["FTM"], "AST": team["AST"], "BLK": team["BLK"], "STL": team["STL"],
        "TOV": team["TOV"], "TOV_team": team["TOV_team"], "DREB": team["DREB"], "OREB": team["OREB"],
        "F_tech": team["F_tech"], "F_personal": team["F_personal"], "team_score": team["team_score"],
        "opponent_team_score": team["opponent_team_score"], "largest_lead": team["largest_lead"],
        "notD1_incomplete": team["notD1_incomplete"], "OT_length_min_tot": team["OT_length_min_tot"],
        "rest_days": team["rest_days"], "attendance": team["attendance"], "tz_dif_H_E": team["tz_dif_H_E"],
        "prev_game_dist": team["prev_game_dist"], "home_away": team["home_away"],
        "home_away_NS": team["home_away_NS"], "travel_dist": team["travel_dist"]
    }

def calculate_strength(stats, weights):
    attack = (stats["FGA_2"] * 2 * 0.5 + stats["FGA_3"] * 3 * 0.5 + stats["FTA"] * 0.5 +
              stats["largest_lead"] * 0.5 + stats["OREB"] + stats["FGM_2"] * 2 * 0.9 +
              stats["FGM_3"] * 3 * 0.9 + stats["FTM"] * 0.9 + stats["AST"])
    defense = (stats["BLK"] * 0.2 + stats["STL"] * 0.4 + stats["TOV"] * 0.2 + stats["TOV_team"] * 0.5 +
               stats["DREB"] * 0.1 - stats["F_personal"] * 0.1 + stats["F_tech"])
    return (defense * weights[1] + attack * weights[0]) * weights[2]

def calculate_reward(predicted, real, strengthA, strengthB):
    calculated_ratio = abs(strengthA / (strengthB + 1e-6))
    actual_ratio = abs(real["team_score"] / real["opponent_team_score"]) if real["opponent_team_score"] != 0 else 1
    ratio_difference = abs(actual_ratio - calculated_ratio)
    return max(0, 2 - ratio_difference) if predicted == (1 if strengthA > strengthB else 2) else 0

def update_q(state, action, reward, new_state):
    Q_table[state][action] += learning_rate * (
        reward + discount_factor * np.max(Q_table[new_state]) - Q_table[state][action]
    )

def normalize_weights(weights):
    return (np.clip(weights, 0.1, 100) / np.sum(weights)) * 100

def get_state(weights):
    return tuple(round(w, 1) for w in weights)

def get_best_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q_table[state])

reward_history = []

for episode in range(episodes):
    state = np.random.randint(0, num_states)
    total_reward = 0

    for step in range(100):
        action = get_best_action(state)
        random_index = np.random.randint(0, len(raw_data))
        teamA = raw_data.iloc[random_index]
        teamB = find_teamB(random_index, raw_data)
        if teamB is None:
            continue

        statsA = extract_team_stats(teamA)
        statsB = extract_team_stats(teamB)
        strengthA = calculate_strength(statsA, weights)
        strengthB = calculate_strength(statsB, weights)

        if action < 3:
            weights[action] += 1
        elif action < 6:
            weights[action - 3] -= 1
        weights = normalize_weights(weights)

        new_state = get_state(weights)
        real_result = 1 if statsA["team_score"] > statsB["team_score"] else 2
        predicted_result = 1 if strengthA > strengthB else 2
        reward = calculate_reward(predicted_result, statsA, strengthA, strengthB)
        update_q(state, action, reward, new_state)

        state = new_state
        total_reward += reward

    reward_history.append((total_reward, weights.copy()))


    if (episode + 1) % 100 == 0:
        last_100_rewards = reward_history[-100:]
        max_reward, max_weights = max(last_100_rewards, key=lambda x: x[0])
        min_reward, min_weights = min(last_100_rewards, key=lambda x: x[0])

        print(f"Episode {episode + 1}")
        print(f"  Max Reward: {max_reward:.2f} - Weights: {max_weights}")
        print(f"  Min Reward: {min_reward:.2f} - Weights: {min_weights}")
