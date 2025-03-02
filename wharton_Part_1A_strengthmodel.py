import numpy as np
import pandas as pd


file_path = "/content/merged_statistics.csv"
raw_data = pd.read_csv(file_path)

results = []

w_attack = 0.4517758898    # Example weight for attack
w_defense = 0.401212117   # Example weight for defense
w_external = 0.1470119925  # Example weight for external factors

for _, team in raw_data.iterrows():
    team_name = team["team"]

    Field_goal_attempt = team["FGA_2"]
    Field_goal_made = team["FGM_2"]
    Three_goal_attempt = team["FGA_3"]
    Three_goal_made = team["FGM_3"]
    Field_throw_attempt = team["FTA"]
    Field_throw_made = team["FTM"]
    Assist = team["AST"]
    Block = team["BLK"]
    Steal = team["STL"]
    Turnover = team["TOV"]
    Team_turnover = team["TOV_team"]
    Defence_rebound = team["DREB"]
    Offence_rebound = team["OREB"]
    Foul_tech = team["F_tech"]
    Foul_pers = team["F_personal"]
    Team_score = team["team_score"]
    Largest_lead = team["largest_lead"]
    Not_D1_incomplete = team["notD1_incomplete"]
    OT_length_min_tot = team["OT_length_min_tot"]
    Rest_days = team["rest_days"]
    Attendance = team["attendance"]
    Time_Zone_Dif = team["tz_dif_H_E"]
    Prev_game_dist = team["prev_game_dist"]
    Home_away_NS = team["home_away_NS"]
    Travel_dist = team["travel_dist"]
    Opposition_Win_Rate = team["avg_opponent_win_rate"]

    attack = (Field_goal_attempt * 2 * 0.5 + Three_goal_attempt * 3 * 0.5 + Field_throw_attempt * 0.5 +
              Largest_lead * 0.5 + Offence_rebound + Field_goal_made * 2 * 0.9 + Three_goal_made * 3 * 0.9 +
              Field_throw_made * 0.9 + Assist)

    defense = (Block * 3 + Steal * 6 + Turnover * 4 + Team_turnover * 7.5 +
           Defence_rebound * 2 - Foul_pers * 1.5 + Foul_tech * 15)

    home = 0.5 if Home_away_NS == "home" else 0

    if Rest_days > 15:
        rest = 0.1
    elif Rest_days > 10:
        rest = 0.2
    elif Rest_days > 7:
        rest = 0.25
    elif Rest_days > 5:
        rest = 0.2
    elif Rest_days > 3:
        rest = 0.1
    else:
        rest = 0

    time = 0.125 if Time_Zone_Dif < 3 else 0

    if Travel_dist < 200:
        travel = 0.125
    elif Travel_dist < 400:
        travel = 0.1
    elif Travel_dist < 800:
        travel = 0.08
    elif Travel_dist < 1200:
        travel = 0.06
    elif Travel_dist < 1500:
        travel = 0.04
    else:
        travel = 0

    performance = home + rest + time + travel
    opponents_strength = 1.0 if Not_D1_incomplete == False else 1.3
    external = 0.6 * opponents_strength + 0.2 * performance + 0.2 * Opposition_Win_Rate

    strength = (w_attack * attack) + (w_defense * defense) + (w_external * external)


    results.append([team_name, strength, external, attack, defense, performance])

columns = ["Team", "Strength", "External Factors", "Attack", "Defense", "Performance Boost"]
results_df = pd.DataFrame(results, columns=columns)

output_path = "TEST3_team_strengths_weighted.csv"
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
