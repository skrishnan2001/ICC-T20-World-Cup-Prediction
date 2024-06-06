import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to preprocess data
def preprocess_data(df, train=True, ref_df=None):
    categorical_columns = ['team_1', 'team_2', 'venue', 'city', 'toss_winner', 'toss_decision', 'player_of_match', 'winner']
    columns_to_encode = [col for col in categorical_columns if col in df.columns]
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    
    if train:
        if 'date' in df.columns:
            df.drop(columns=['date'], inplace=True)
        if 'season' in df.columns:
            df.drop(columns=['season'], inplace=True)
    else:
        for col in ref_df.columns:
            if col not in df.columns:
                df[col] = 0
        df = df[ref_df.columns]
    
    return df

# Load processed data
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Processed DataFrame shape:", df.shape)
    print("Processed DataFrame columns:", df.columns)
    print("First few rows of the processed DataFrame:\n", df.head())
    return df

# Handle 'win' column
def handle_win_column(df):
    if 'win' in df.columns:
        df['win'] = pd.to_numeric(df['win'], errors='coerce')
        df.dropna(subset=['win'], inplace=True)
    else:
        raise ValueError("The 'win' column is missing from the processed data")
    print("Shape after handling 'win' column:", df.shape)
    if df.empty:
        raise ValueError("The processed DataFrame is empty after handling 'win' column")
    return df

# Split data into training and testing sets
def split_data(df):
    X = df.drop(columns=['win'])
    y = df['win'].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Predict the outcomes of the fixtures
def predict_match_outcome(fixtures_df, model, processed_df):
    predictions = []
    probabilities = []
    
    for _, row in fixtures_df.iterrows():
        match_features = {
            'team_1': row['team_1'],
            'team_2': row['team_2'],
            'venue': row['venue'],
            'city': row['city'],
            'toss_winner': row['toss_winner'],
            'toss_decision': row['toss_decision'],
            'season': row['season']
        }
        match_df = pd.DataFrame([match_features])
        match_df = preprocess_data(match_df, train=False, ref_df=processed_df)
        match_df = match_df.drop(columns=['win'])
        
        prediction = model.predict(match_df)
        prediction_proba = model.predict_proba(match_df)
        
        predictions.append(prediction[0])
        probabilities.append(prediction_proba[0])
    
    fixtures_df['predicted_winner'] = predictions
    fixtures_df['team_1_win_probability'] = [prob[0] for prob in probabilities]
    fixtures_df['team_2_win_probability'] = [prob[1] for prob in probabilities]
    
    return fixtures_df

# Simulate the tournament
def simulate_tournament(fixtures_df):
    teams = set(fixtures_df['team_1']).union(set(fixtures_df['team_2']))
    team_wins = {team: 0 for team in teams}
    
    for _, row in fixtures_df.iterrows():
        winner = row['predicted_winner']
        team_wins[winner] += 1
    
    tournament_winner = max(team_wins, key=team_wins.get)
    return tournament_winner

# Main function to run the script
def main():
    processed_data_file = './data/processed_data/processed_t20_data.csv'
    processed_df = load_data(processed_data_file)
    processed_df = handle_win_column(processed_df)
    processed_df = preprocess_data(processed_df, train=True)
    
    X_train, X_test, y_train, y_test = split_data(processed_df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    fixtures_file = './data/fixtures/fixture_T20_world_cup_2024.csv'
    fixtures_df = pd.read_csv(fixtures_file)
    
    predicted_fixtures_df = predict_match_outcome(fixtures_df, model, processed_df)
    results_file = 'predicted_match_results.csv'
    predicted_fixtures_df.to_csv(results_file, index=False)
    
    tournament_winner = simulate_tournament(predicted_fixtures_df)
    print(f'The predicted winner of the 2024 T20 Cricket World Cup is: {tournament_winner}')
    
    tournament_winner_file = './data/predictions/tournament_winner.csv'
    tournament_winner_df = pd.DataFrame({'tournament_winner': [tournament_winner]})
    tournament_winner_df.to_csv(tournament_winner_file, index=False)

if __name__ == "__main__":
    main()
