from xgboost_class import xgboost_model
import os
from dotenv import load_dotenv
load_dotenv()


def train_model_and_plot(train_data, train_label, params, model_name, tag, log_model, model_path):
    results = xgml.train_model(train_data, train_label, params)

    xgml.plot_roc(results, os.getenv(
        "XGB_IMAGE_PATH") + model_name + '_roc.png')

    xgml.plot_goal_rate(results, os.getenv(
        "XGB_IMAGE_PATH") + model_name + '_goal_rate.png')

    xgml.plot_cumulative(results, os.getenv(
        "XGB_IMAGE_PATH") + model_name + '_cumulative.png')

    xgml.plot_calibration_curve(results, os.getenv(
        "XGB_IMAGE_PATH") + model_name + '_calibration_curve.png')

    xgml.log_experiment(results, tag, log_model, model_name, model_path)


if __name__ == "__main__":
    xgml = xgboost_model()

    train_data, train_label = xgml.baseline_preprocessing()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
    }

    print('Training baseline model and plotting...')
    train_model_and_plot(train_data, train_label, params,
                         'baseline', 'xgboost_baseline', False, '')

    param_grid = {
        'objective': ['binary:logistic'],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.01],
        'reg_alpha': [0.1, 0.001]
    }

    print('Preprocessing data for hyperparameter tuning...')
    train_data, train_label = xgml.preprocessing()

    print('Performing hyperparameter tuning...')
    best_params, best_score = xgml.hyper_parameter_tuning(
        train_data, train_label, param_grid)

    best_params['eval_metric'] = 'logloss'
    print(best_params)
    del best_params['n_estimators']

    print('Training hyperparameter tuned model without cross validation for saving...')
    hptuned_model = xgml.train_model_nocv(train_data, train_label, best_params)

    print('Saving hyperparameter tuned model...')
    hptuned_model.save_model(
        os.getenv("XGB_MODEL_PATH") + 'hptuned_model.json')

    print('Training hyperparameter tuned model and plotting...')
    train_model_and_plot(train_data, train_label, best_params,
                         'hp_tuning', 'xgboost_hp_tuning', True,
                         os.getenv("XGB_MODEL_PATH") + 'hptuned_model.json')

    print('Performing feature selection...')
    train_data_feature_selected = xgml.feature_selection(train_data, train_label, best_params, os.getenv(
        "XGB_IMAGE_PATH") + 'feature_importance.png', os.getenv("XGB_IMAGE_PATH") + 'shap_importance.png')

    print('Training feature selected model without cross validation for saving...')
    feature_selected_model = xgml.train_model_nocv(
        train_data_feature_selected, train_label, best_params)

    print('Saving feature selected model...')
    feature_selected_model.save_model(
        os.getenv("XGB_MODEL_PATH") + 'feature_selected_model.json')

    print('Training feature selected model and plotting...')
    train_model_and_plot(train_data_feature_selected,
                         train_label, best_params, 'feature_selected', 'xgboost_feature_selected',
                         True, os.getenv("XGB_MODEL_PATH") + 'feature_selected_model.json')
