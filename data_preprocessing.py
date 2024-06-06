import os
import pandas as pd
import numpy as np

# Parsing functions
def parse_info_file(file_path):
    info_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                key = parts[1].strip()
                value = ','.join(parts[2:]).strip()
                if key in info_dict:
                    if isinstance(info_dict[key], list):
                        info_dict[key].append(value)
                    else:
                        info_dict[key] = [info_dict[key], value]
                else:
                    info_dict[key] = value
    return info_dict

def parse_match_file(file_path):
    return pd.read_csv(file_path, dtype=str, low_memory=False)

# Normalization functions
def normalize_info_dict(info_dict):
    normalized = {}
    attributes = [
        'team', 'gender', 'season', 'date', 'venue', 'city',
        'toss_winner', 'toss_decision', 'player_of_match',
        'umpire', 'tv_umpire', 'match_referee', 'winner', 'winner_runs'
    ]

    for key in attributes:
        if key in info_dict:
            value = info_dict[key]
            if isinstance(value, list):
                normalized[key] = ', '.join(value)
            else:
                normalized[key] = value

    return normalized

# Combination functions
def combine_info_files(data_dir, output_file):
    info_files = [f for f in os.listdir(data_dir) if f.endswith('_info.csv')]
    all_info_data = []

    for info_file in info_files:
        file_path = os.path.join(data_dir, info_file)
        parsed_info = parse_info_file(file_path)
        normalized_info = normalize_info_dict(parsed_info)

        if 'team' in normalized_info:
            teams = normalized_info['team'].split(', ')
            normalized_info['team_1'] = teams[0].strip()
            normalized_info['team_2'] = teams[1].strip() if len(teams) > 1 else 'Unknown'
            del normalized_info['team']

        if 'winner_runs' in normalized_info:
            normalized_info['win'] = normalized_info['winner_runs'].strip()
            del normalized_info['winner_runs']
        else:
            normalized_info['win'] = 'Unknown'

        all_info_data.append(normalized_info)

    info_df = pd.DataFrame(all_info_data)

    info_df['date'] = pd.to_datetime(info_df['date'], errors='coerce').dt.strftime('%Y/%m/%d')

    required_columns = ['team_1', 'team_2', 'season', 'date', 'venue', 'city', 'toss_winner', 'toss_decision', 'player_of_match', 'winner', 'win']
    for col in required_columns:
        if col not in info_df.columns:
            info_df[col] = 'Unknown'

    info_df = info_df[required_columns]
    info_df.fillna('Unknown', inplace=True)

    info_df.to_csv(output_file, index=False)
    print(f'Combined info CSV file saved as {output_file}')

def combine_match_files(data_dir, output_file):
    match_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.endswith('_info.csv')]
    all_match_data = []

    for match_file in match_files:
        file_path = os.path.join(data_dir, match_file)
        match_df = parse_match_file(file_path)
        match_df['match_id'] = match_file.split('.')[0]
        all_match_data.append(match_df)

    combined_match_df = pd.concat(all_match_data, ignore_index=True)
    combined_match_df.to_csv(output_file, index=False)
    print(f'Combined match CSV file saved as {output_file}')

# Preprocessing and feature engineering
def preprocess_and_feature_engineer(info_file, match_file, output_file):
    info_df = pd.read_csv(info_file)
    match_df = pd.read_csv(match_file, dtype=str, low_memory=False)

    info_df.fillna('Unknown', inplace=True)

    categorical_columns = ['team_1', 'team_2', 'venue', 'city', 'toss_winner', 'toss_decision', 'player_of_match', 'winner']
    for col in categorical_columns:
        info_df[col] = info_df[col].astype(str)

    # Create new features
    team_stats = match_df.groupby(['batting_team', 'bowling_team']).size().reset_index(name='matches_played')
    team_stats['total_matches'] = team_stats.groupby('batting_team')['matches_played'].transform('sum')

    info_df = pd.merge(info_df, team_stats, how='left', left_on=['team_1', 'team_2'], right_on=['batting_team', 'bowling_team'])

    info_df.fillna(0, inplace=True)

    info_df = pd.get_dummies(info_df, columns=categorical_columns, drop_first=True)

    numerical_columns = ['matches_played', 'total_matches']
    info_df[numerical_columns] = (info_df[numerical_columns] - info_df[numerical_columns].mean()) / info_df[numerical_columns].std()

    info_df.to_csv(output_file, index=False)
    print(f'Processed and feature-engineered data saved as {output_file}')

# Main function to execute the workflow
def main():
    data_dir = './data/t20s_male_csv2'
    info_output_file = './data/processed_data/match_info_combined.csv'
    match_output_file = './data/processed_data/ball_by_ball_combined.csv'
    processed_output_file = './data/processed_data/processed_t20_data.csv'

    combine_info_files(data_dir, info_output_file)
    combine_match_files(data_dir, match_output_file)
    preprocess_and_feature_engineer(info_output_file, match_output_file, processed_output_file)

if __name__ == "__main__":
    main()
