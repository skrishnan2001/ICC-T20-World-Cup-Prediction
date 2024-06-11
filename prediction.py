import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

class CricketWorldCupSimulator:
    def __init__(self, ball_by_ball_path, match_info_path, fixture_path):
        self.ball_by_ball_path = ball_by_ball_path
        self.match_info_path = match_info_path
        self.fixture_path = fixture_path

    def load_data(self):
        dtype_spec = {
            'wides': 'float64', 'noballs': 'float64', 'byes': 'float64', 
            'legbyes': 'float64', 'penalty': 'float64'
        }
        self.ball_by_ball_df = pd.read_csv(self.ball_by_ball_path, dtype=dtype_spec, low_memory=False)
        self.match_info_df = pd.read_csv(self.match_info_path)
        self.fixture_df = pd.read_csv(self.fixture_path)

    def preprocess_data(self):
        self._convert_dates()
        self._fill_missing_values()
        self._standardize_team_names()
        self.team_stats = self._calculate_team_stats()
        self.match_stats = self._prepare_training_data()

    def train_model(self):
        features, target = self._get_features_and_target()
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        self.accuracy = accuracy_score(y_val, y_pred)
        self.conf_matrix = confusion_matrix(y_val, y_pred)

        # if self.accuracy < 0.80:
        #     raise Exception(f"Model accuracy is below 80%. Current accuracy: {self.accuracy:.2f}")

    def simulate_world_cup(self):
        fixture_stats = self._prepare_fixture_data()
        fixture_predictions = self.model.predict(fixture_stats)
        self.fixture_df['prediction'] = fixture_predictions

        group_winners = self.fixture_df.groupby('Group').apply(lambda x: x.loc[x['prediction'].idxmax(), 'Team-A'])
        self.world_cup_winner = group_winners.mode()[0]

    def _convert_dates(self):
        self.ball_by_ball_df['start_date'] = pd.to_datetime(self.ball_by_ball_df['start_date'], errors='coerce')
        self.match_info_df['date'] = pd.to_datetime(self.match_info_df['date'], errors='coerce')
        self.fixture_df['Date'] = pd.to_datetime(self.fixture_df['Date'], format='%d-%b-%y', errors='coerce')

    def _fill_missing_values(self):
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        columns_to_fill = ['wides', 'noballs', 'byes', 'legbyes', 'penalty']
        self.ball_by_ball_df[columns_to_fill] = imputer.fit_transform(self.ball_by_ball_df[columns_to_fill])
        self.ball_by_ball_df['wicket_type'].fillna('none', inplace=True)
        self.ball_by_ball_df['player_dismissed'].fillna('none', inplace=True)

    def _standardize_team_names(self):
        def standardize_names(df, columns):
            for col in columns:
                df[col] = df[col].str.lower().str.strip()
            return df
        
        self.ball_by_ball_df = standardize_names(self.ball_by_ball_df, ['batting_team', 'bowling_team'])
        self.match_info_df = standardize_names(self.match_info_df, ['team_1', 'team_2', 'toss_winner', 'winner'])
        self.fixture_df = standardize_names(self.fixture_df, ['Team-A', 'Team-B'])

    def _calculate_team_stats(self):
        team_stats = self.ball_by_ball_df.groupby('batting_team').agg({
            'runs_off_bat': 'sum',
            'ball': 'count',
            'wides': 'sum',
            'noballs': 'sum',
            'byes': 'sum',
            'legbyes': 'sum',
            'wicket_type': lambda x: (x != 'none').count()
        }).reset_index()

        team_stats.columns = ['team', 'total_runs', 'total_balls', 'total_wides', 'total_noballs', 'total_byes', 'total_legbyes', 'total_wickets']
        team_stats['average_runs'] = team_stats['total_runs'] / team_stats['total_balls'] * 6
        team_stats['average_wickets'] = team_stats['total_wickets'] / team_stats['total_balls'] * 6
        return team_stats

    def _prepare_training_data(self):
        match_info_df = self.match_info_df.rename(columns={'team_1': 'team', 'team_2': 'opponent'})
        match_stats = match_info_df.merge(self.team_stats, on='team', how='left')
        match_stats = match_stats.rename(columns={col: 'team_' + col for col in self.team_stats.columns if col != 'team'})

        match_stats = match_stats.merge(self.team_stats, left_on='opponent', right_on='team', suffixes=('', '_opponent'))
        match_stats = match_stats.rename(columns={col: 'opponent_' + col for col in self.team_stats.columns if col != 'team'})

        print("Match Stats Columns after merging and renaming:", match_stats.columns)

        # Ensure columns to drop are within the dataframe
        columns_to_drop = [col for col in ['team_opponent'] if col in match_stats.columns]
        match_stats.drop(columns=columns_to_drop, inplace=True)

        match_stats.fillna(0, inplace=True)

        if 'winner' not in match_stats.columns or 'team' not in match_stats.columns:
            raise KeyError("'winner' or 'team' column not found in match_stats")

        match_stats['target'] = (match_stats['winner'] == match_stats['team']).astype(int)
        return match_stats

    def _get_features_and_target(self):
        features = [col for col in self.match_stats.columns if col.startswith('team_') or col.startswith('opponent_')]
        target = self.match_stats['target']
        return self.match_stats[features], target

    def _prepare_fixture_data(self):
        fixture_df = self.fixture_df.rename(columns={'Team-A': 'team', 'Team-B': 'opponent'})
        fixture_stats = fixture_df.merge(self.team_stats, on='team', how='left')
        fixture_stats = fixture_stats.rename(columns={col: 'team_' + col for col in self.team_stats.columns if col != 'team'})

        fixture_stats = fixture_stats.merge(self.team_stats, left_on='opponent', right_on='team', suffixes=('', '_opponent'))
        fixture_stats = fixture_stats.rename(columns={col: 'opponent_' + col for col in self.team_stats.columns if col != 'team'})
        columns_to_drop = [col for col in ['team_opponent', 'team'] if col in fixture_stats.columns]
        fixture_stats.drop(columns=columns_to_drop, inplace=True)
        
        fixture_stats.fillna(0, inplace=True)
        
        return fixture_stats[[col for col in fixture_stats.columns if col.startswith('team_') or col.startswith('opponent_')]]

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.simulate_world_cup()
        return self.world_cup_winner, self.accuracy, self.conf_matrix

if __name__ == "__main__":
    simulator = CricketWorldCupSimulator(
        ball_by_ball_path='./data/processed_data/ball_by_ball_combined.csv',
        match_info_path='./data/processed_data/match_info_combined.csv',
        fixture_path='./data/fixtures/fixture_T20_world_cup_2024.csv'
    )

    try:
        winner, accuracy, conf_matrix = simulator.run()
        print(f"Predicted World Cup Winner: {winner}")
        print(f"Model Accuracy: {accuracy:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
    except Exception as e:
        print(e)
