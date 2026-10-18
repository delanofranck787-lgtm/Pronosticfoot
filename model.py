import numpy as np
import pandas as pd
from scipy.stats import poisson


class DixonColesModel:
    """
    Modèle de prédiction football basé sur la distribution de Poisson
    avec ajustement de force attaque/défense par équipe.
    Simple, robuste, éprouvé.
    """

    def __init__(self):
        self.team_attack = {}
        self.team_defense = {}
        self.home_advantage = 1.0
        self.league_avg_home = 1.4
        self.league_avg_away = 1.1

    def fit(self, df: pd.DataFrame):
        """Entraîne le modèle sur l'historique des matchs."""
        if df is None or len(df) < 10:
            return False

        df = df.copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        self.league_avg_home = df["home_goals"].mean()
        self.league_avg_away = df["away_goals"].mean()

        # Force d'attaque et défense par équipe (ratio vs moyenne)
        teams = pd.concat([df["home"], df["away"]]).unique()

        for team in teams:
            home_games = df[df["home"] == team]
            away_games = df[df["away"] == team]

            goals_scored = home_games["home_goals"].sum() + away_games["away_goals"].sum()
            goals_conceded = home_games["away_goals"].sum() + away_games["home_goals"].sum()
            n_games = len(home_games) + len(away_games)

            if n_games == 0:
                self.team_attack[team] = 1.0
                self.team_defense[team] = 1.0
            else:
                avg_scored = goals_scored / n_games
                avg_conceded = goals_conceded / n_games
                league_avg = (self.league_avg_home + self.league_avg_away) / 2
                self.team_attack[team] = avg_scored / league_avg if league_avg > 0 else 1.0
                self.team_defense[team] = avg_conceded / league_avg if league_avg > 0 else 1.0

        return True

    def _lambdas(self, home_team, away_team):
        attack_home = self.team_attack.get(home_team, 1.0)
        defense_home = self.team_defense.get(home_team, 1.0)
        attack_away = self.team_attack.get(away_team, 1.0)
        defense_away = self.team_defense.get(away_team, 1.0)

        lambda_home = attack_home * defense_away * self.league_avg_home
        lambda_away = attack_away * defense_home * self.league_avg_away
        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def predict(self, home_team, away_team, max_goals=6):
        """Retourne les probabilités 1X2, over/under, BTTS, score exact."""
        lam_h, lam_a = self._lambdas(home_team, away_team)

        # Matrice des scores possibles
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                matrix[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)

        prob_home = np.sum(np.tril(matrix, -1))  # victoire dom
        prob_draw = np.sum(np.diag(matrix))       # nul
        prob_away = np.sum(np.triu(matrix, 1))    # victoire ext

        # Normalisation
        total = prob_home + prob_draw + prob_away
        prob_home, prob_draw, prob_away = prob_home/total, prob_draw/total, prob_away/total

        # Over/Under 2.5
        over_25 = 0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i + j > 2.5:
                    over_25 += matrix[i, j]

        # BTTS (les deux marquent)
        btts = 1 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]

        # Score le plus probable
        idx = np.unravel_index(np.argmax(matrix), matrix.shape)

        return {
            "prob_home": float(prob_home),
            "prob_draw": float(prob_draw),
            "prob_away": float(prob_away),
            "over_25": float(over_25),
            "under_25": float(1 - over_25),
            "btts": float(btts),
            "lambda_home": float(lam_h),
            "lambda_away": float(lam_a),
            "most_likely_score": f"{idx[0]}-{idx[1]}",
            "score_probability": float(matrix[idx]),
                 }
