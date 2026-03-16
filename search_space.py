search_space = {

    "seed": [0, 42, 123],

    "use_shooting": [True, False],
    "use_free_throws": [True, False],
    "use_rebounding": [True, False],
    "use_turnovers": [True, False],
    "use_defense": [True, False],
    "use_playmaking": [True, False],
    "use_efficiency": [True, False],
    "use_win_history": [True, False],

    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5],

    "sample_size": [10000, 15000, 20000]
}