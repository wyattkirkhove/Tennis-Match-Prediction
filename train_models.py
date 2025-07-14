"""
Tennis Match Prediction Model Training Pipeline

This script loads engineered features from data/final_features.csv, splits the data into training and test sets,
performs preprocessing, loads best hyperparameters, trains multiple models (with calibration and stacking),
and evaluates their performance. Results and predictions are saved to disk.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
from sklearn.impute import SimpleImputer


def load_best_params(filename='models/best_hyperparameters.json'):
    """
    Load the best hyperparameters from a JSON file.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def load_and_split_data(data_path, target='RESULT', test_size=0.2, random_state=42):
    """
    Load the full feature dataset and split into train and test sets.
    """
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) == 2 else None
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, df


def prepare_features(X_train, X_test, features):
    """
    Prepare features for both training and test data using a preprocessing pipeline.
    """
    categorical_features = ['surface', 'round', 'tourney_level']
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    X_train_processed = preprocessor.fit_transform(X_train[features])
    X_test_processed = preprocessor.transform(X_test[features])
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    return X_train_processed, X_test_processed, preprocessor


def load_existing_model(model_name):
    """Load an existing model if it exists."""
    model_path = f'models/{model_name.lower().replace(" ", "_")}_model.joblib'
    platt_path = f'models/{model_name.lower().replace(" ", "_")}_platt_model.joblib'
    isotonic_path = f'models/{model_name.lower().replace(" ", "_")}_isotonic_model.joblib'
    if os.path.exists(model_path) and os.path.exists(platt_path) and os.path.exists(isotonic_path):
        print(f"Loading existing {model_name} model")
        return (
            joblib.load(model_path),
            joblib.load(platt_path),
            joblib.load(isotonic_path)
        )
    return None


def train_and_evaluate_models(X_train, X_test, y_train, y_test, features, original_X_test):
    """
    Train models, calibrate, stack, and evaluate. Save predictions and print results.
    """
    results = {}
    test_predictions = {}
    best_models = {}
    calibrated_models = {}

    best_params = load_best_params()
    if best_params is None:
        raise ValueError("No saved hyperparameters found. Please run the random search first.")

    print("\nTraining models with saved best parameters")
    for name, params in best_params.items():
        existing_models = load_existing_model(name)
        if existing_models is not None:
            base_model, platt_model, isotonic_model = existing_models
            best_models[name] = base_model
            calibrated_models[f"{name}_platt"] = platt_model
            calibrated_models[f"{name}_isotonic"] = isotonic_model
            continue
        print(f"Training new {name} model")
        if name == 'XGBoost':
            base_model = xgb.XGBClassifier(**params, random_state=42)
        elif name == 'LightGBM':
            base_model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        elif name == 'Logistic Regression':
            base_model = LogisticRegression(**params, random_state=42)
        else:
            continue
        base_model.fit(X_train, y_train)
        best_models[name] = base_model
        print(f"Calibrating {name} probabilities")
        platt_model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
        platt_model.fit(X_train, y_train)
        calibrated_models[f"{name}_platt"] = platt_model
        isotonic_model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
        isotonic_model.fit(X_train, y_train)
        calibrated_models[f"{name}_isotonic"] = isotonic_model
        joblib.dump(base_model, f'models/{name.lower().replace(" ", "_")}_model.joblib')
        joblib.dump(platt_model, f'models/{name.lower().replace(" ", "_")}_platt_model.joblib')
        joblib.dump(isotonic_model, f'models/{name.lower().replace(" ", "_")}_isotonic_model.joblib')
        y_pred_proba_base = base_model.predict_proba(X_test)[:, 1]
        y_pred_proba_platt = platt_model.predict_proba(X_test)[:, 1]
        y_pred_proba_isotonic = isotonic_model.predict_proba(X_test)[:, 1]
        test_predictions[f"{name}_base"] = y_pred_proba_base
        test_predictions[f"{name}_platt"] = y_pred_proba_platt
        test_predictions[f"{name}_isotonic"] = y_pred_proba_isotonic
        # Evaluate on train set
        y_pred_proba_base_train = base_model.predict_proba(X_train)[:, 1]
        y_pred_proba_platt_train = platt_model.predict_proba(X_train)[:, 1]
        y_pred_proba_isotonic_train = isotonic_model.predict_proba(X_train)[:, 1]
        results[name] = {
            'base': {
                'train': {
                    'accuracy': accuracy_score(y_train, (y_pred_proba_base_train > 0.5).astype(int)),
                    'brier': brier_score_loss(y_train, y_pred_proba_base_train),
                    'roc_auc': roc_auc_score(y_train, y_pred_proba_base_train)
                },
                'test': {
                    'accuracy': accuracy_score(y_test, (y_pred_proba_base > 0.5).astype(int)),
                    'brier': brier_score_loss(y_test, y_pred_proba_base),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba_base)
                }
            },
            'platt': {
                'train': {
                    'accuracy': accuracy_score(y_train, (y_pred_proba_platt_train > 0.5).astype(int)),
                    'brier': brier_score_loss(y_train, y_pred_proba_platt_train),
                    'roc_auc': roc_auc_score(y_train, y_pred_proba_platt_train)
                },
                'test': {
                    'accuracy': accuracy_score(y_test, (y_pred_proba_platt > 0.5).astype(int)),
                    'brier': brier_score_loss(y_test, y_pred_proba_platt),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba_platt)
                }
            },
            'isotonic': {
                'train': {
                    'accuracy': accuracy_score(y_train, (y_pred_proba_isotonic_train > 0.5).astype(int)),
                    'brier': brier_score_loss(y_train, y_pred_proba_isotonic_train),
                    'roc_auc': roc_auc_score(y_train, y_pred_proba_isotonic_train)
                },
                'test': {
                    'accuracy': accuracy_score(y_test, (y_pred_proba_isotonic > 0.5).astype(int)),
                    'brier': brier_score_loss(y_test, y_pred_proba_isotonic),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba_isotonic)
                }
            }
        }
        print(f"\n{name} Results:")
        for variant in ['base', 'platt', 'isotonic']:
            print(f"\n{variant.capitalize()} Model:")
            print(f"Train  - Accuracy: {results[name][variant]['train']['accuracy']:.4f}, Brier: {results[name][variant]['train']['brier']:.4f}, ROC AUC: {results[name][variant]['train']['roc_auc']:.4f}")
            print(f"Test   - Accuracy: {results[name][variant]['test']['accuracy']:.4f}, Brier: {results[name][variant]['test']['brier']:.4f}, ROC AUC: {results[name][variant]['test']['roc_auc']:.4f}")

    print("\nTraining Stacking Ensemble with best parameters")
    base_estimators = []
    for name, model in best_models.items():
        base_estimators.extend([
            (f"{name.lower().replace(' ', '_')}_base", model),
            (f"{name.lower().replace(' ', '_')}_platt", calibrated_models[f"{name}_platt"]),
            (f"{name.lower().replace(' ', '_')}_isotonic", calibrated_models[f"{name}_isotonic"])
        ])
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    X_train_np = X_train.to_numpy(copy=True)
    X_test_np = X_test.to_numpy(copy=True)
    y_train_np = y_train.to_numpy(copy=True)
    stacking_clf.fit(X_train_np, y_train_np)
    stacking_pred_proba = stacking_clf.predict_proba(X_test_np)[:, 1]
    test_predictions['Stacking'] = stacking_pred_proba
    # Stacking ensemble train/test comparison
    stacking_pred_proba_train = stacking_clf.predict_proba(X_train_np)[:, 1]
    results['Stacking'] = {
        'train': {
            'accuracy': accuracy_score(y_train, (stacking_pred_proba_train > 0.5).astype(int)),
            'brier': brier_score_loss(y_train, stacking_pred_proba_train),
            'roc_auc': roc_auc_score(y_train, stacking_pred_proba_train)
        },
        'test': {
            'accuracy': accuracy_score(y_test, (stacking_pred_proba > 0.5).astype(int)),
            'brier': brier_score_loss(y_test, stacking_pred_proba),
            'roc_auc': roc_auc_score(y_test, stacking_pred_proba)
        }
    }
    print("\nStacking Ensemble Results:")
    print(f"Train  - Accuracy: {results['Stacking']['train']['accuracy']:.4f}, Brier: {results['Stacking']['train']['brier']:.4f}, ROC AUC: {results['Stacking']['train']['roc_auc']:.4f}")
    print(f"Test   - Accuracy: {results['Stacking']['test']['accuracy']:.4f}, Brier: {results['Stacking']['test']['brier']:.4f}, ROC AUC: {results['Stacking']['test']['roc_auc']:.4f}")

    # Save the stacking model
    joblib.dump(stacking_clf, 'models/stacking_model.joblib')
    
    test_results = original_X_test.reset_index(drop=True).copy()
    test_results['RESULT'] = y_test.values

    for name, probs in test_predictions.items():
        test_results[f'{name.lower().replace(" ", "_")}_win_probability'] = probs
    output_file = 'data/test_predictions.csv'
    test_results.to_csv(output_file, index=False)
    print(f"\nTest predictions saved to {output_file}")

    # Print summary
    print("\nModel Comparison Summary:")
    print("-" * 120)
    print(f"{'Model':<30} {'Train Acc':<10} {'Test Acc':<10} {'Train Brier':<12} {'Test Brier':<12} {'Train ROC AUC':<14} {'Test ROC AUC':<14}")
    print("-" * 120)
    for name, metrics in results.items():
        if name != 'Stacking':
            for variant in ['base', 'platt', 'isotonic']:
                print(f"{name + ' (' + variant.capitalize() + ')':<30} {metrics[variant]['train']['accuracy']:<10.4f} {metrics[variant]['test']['accuracy']:<10.4f} {metrics[variant]['train']['brier']:<12.4f} {metrics[variant]['test']['brier']:<12.4f} {metrics[variant]['train']['roc_auc']:<14.4f} {metrics[variant]['test']['roc_auc']:<14.4f}")
        else:
            print(f"{name:<30} {metrics['train']['accuracy']:<10.4f} {metrics['test']['accuracy']:<10.4f} {metrics['train']['brier']:<12.4f} {metrics['test']['brier']:<12.4f} {metrics['train']['roc_auc']:<14.4f} {metrics['test']['roc_auc']:<14.4f}")
    return results, best_models, calibrated_models


def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    # Define features
    features = [
        'surface', 'round', 'tourney_level', 'p1_rank', 'p2_rank', 'rank_diff',
        'p1_rank_points', 'p2_rank_points', 'p1_rank_points_diff', 'p1_ht', 'p2_ht',
        'height_diff', 'p1_age', 'p2_age', 'age_diff', 'p1_elo_pre', 'p2_elo_pre',
        'elo_diff', 'p1_surface_elo_pre', 'p2_surface_elo_pre', 'elo_diff_surface',
        '1stIn_pct_diff_last_5', '1stWon_pct_diff_last_5', '2ndWon_pct_diff_last_5',
        'bpSaved_pct_diff_last_5', 'p1_bpFaced_per_match_last_5', 'p2_bpFaced_per_match_last_5',
        'bpConverted_pct_diff_last_5', 'p1_bpCreated_per_match_last_5', 'p2_bpCreated_per_match_last_5',
        'ace_per_match_diff_last_5', 'df_per_match_diff_last_5', 'p1_recent_winrate_last_5',
        'p2_recent_winrate_last_5', 'winrate_diff_last_5', 'elo_growth_diff_5',
        'surface_winrate_diff_last_5', 'p1_recent_return_points_won_last_5',
        'p2_recent_return_points_won_last_5', 'return_points_won_diff_last_5',
        'p1_recent_1st_return_points_won_last_5', 'p2_recent_1st_return_points_won_last_5',
        '1st_return_points_won_diff_last_5', 'p1_recent_return_pts_won_pct_last_5',
        'p2_recent_return_pts_won_pct_last_5', 'return_pts_won_pct_diff_last_5',
        'p1_recent_return_1st_win_pct_last_5', 'p2_recent_return_1st_win_pct_last_5',
        'return_1st_win_pct_diff_last_5', 'p1_recent_return_2nd_win_pct_last_5',
        'p2_recent_return_2nd_win_pct_last_5', 'return_2nd_win_pct_diff_last_5',
        'h2h_wins_p1', 'h2h_wins_p2', 'total_h2h_matches', 'has_h2h_history',
        'h2h_winrate_p1', 'p1_winrate_trend_100_10', 'p2_winrate_trend_100_10',
        'p1_1stWon_trend_100_20', 'p2_1stWon_trend_100_20', 'p1_2ndWon_trend_100_20',
        'p2_2ndWon_trend_100_20', 'p1_bpFaced_trend_100_20', 'p2_bpFaced_trend_100_20',
        'p1_1st_return_points_won_trend_100_20', 'p2_1st_return_points_won_trend_100_20',
        'p1_2nd_return_points_won_trend_100_20', 'p2_2nd_return_points_won_trend_100_20',
        'winrate_trend_diff_100_10', '1stWon_trend_diff_100_20', '2ndWon_trend_diff_100_20',
        'bpFaced_trend_diff_100_20', '1st_return_points_won_trend_diff_100_20',
        'winrate_momentum_diff', 'surface_momentum_diff', 'age_x_elo',
        'p1_serve_return_ratio', 'p2_serve_return_ratio', 'serve_return_ratio_diff',
        'p1_surface_bias', 'p2_surface_bias', 'surface_bias_diff'
    ]
    # Load and split data
    X_train, X_test, y_train, y_test, _ = load_and_split_data('data/final_features.csv', target='RESULT')
    # Prepare features
    X_train_processed, X_test_processed, preprocessor = prepare_features(X_train, X_test, features)
    # Convert processed arrays back to DataFrames with correct feature names
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())
    # Train models and generate predictions
    train_and_evaluate_models(X_train_processed_df, X_test_processed_df, y_train, y_test, features, X_test)


if __name__ == "__main__":
    main() 