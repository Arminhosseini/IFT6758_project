import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from comet_ml import Experiment
from sklearn.model_selection import RepeatedKFold, GridSearchCV
import matplotlib.ticker as mtick
import shap
import functools
import operator
from xgboost import plot_importance
from sklearn.calibration import calibration_curve
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
load_dotenv()
# ref for repeated k_fold and ROC plot:
# https://towardsdatascience.com/pooled-roc-with-xgboost-and-plotly-553a8169680c#:~:text=To%20get%20a%20ROC%20curve,validation%20and%20got%20500%20results.


class xgboost_model():
    def __init__(self) -> None:
        return None

    def baseline_preprocessing(self):
        df = pd.read_csv(os.getenv("MODEL_DATA_PATH"), index_col=0)
        df = df.dropna()
        train_label = df['isgoal']
        train_data = df.loc[:, df.columns.isin(['shot_distance', 'angle'])]
        return train_data, train_label

    def preprocessing(self):
        df = pd.read_csv(os.getenv("MODEL_DATA_PATH"), index_col=0)
        df = df.dropna()
        train_label = df['isgoal']
        train_data = df.drop(['isgoal', 'attackingSide', 'periodTime'], axis=1)
        train_data['is_rebound'].replace({False: 0, True: 1}, inplace=True)
        train_data = pd.get_dummies(train_data, columns=[
                                    'last_event_type', 'shotType'], dtype=np.int32, drop_first=True)
        train_data_min = train_data.min()
        train_data_max = train_data.max()
        train_data = (train_data - train_data_min) / \
            (train_data_max - train_data_min)  # min-max scaling

        return train_data, train_label

    def feature_selection(self, train_data, train_label, model_params, plt_importance_save_path, shap_importance_save_path):
        corr_matrix = train_data.corr()
        corrrelated_features = []
        for column in corr_matrix.columns:
            for row in corr_matrix.index:
                if column != row and np.abs(corr_matrix.loc[row, column]) > 0.9:
                    if column and row not in corrrelated_features:
                        corrrelated_features.append(column)

        train_data = train_data.drop(columns=corrrelated_features, axis=1)
        model = xgb.XGBClassifier(**model_params)
        model.fit(train_data, train_label)
        # ref: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
        ax = plot_importance(model)
        fig = ax.figure
        fig.set_size_inches(20, 20)
        plt.savefig(plt_importance_save_path)
        plt.show()
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(train_data)
        shap.summary_plot(shap_values, train_data, plot_type="bar", show=False)
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(20, 20)
        plt.tight_layout()
        plt.savefig(shap_importance_save_path)
        plt.show()
        # shap.summary_plot(shap_values, train_data, plot_type="bar")
        train_data = train_data.loc[:, train_data.columns.isin(['shot_distance', 'time_last_event', 'angle', 'y-coordinate', 'Change in shot angle',
                                                                'distance_last_event', 'game_second', 'Speed', 'coor_y_last_event', 'x-coordinate',
                                                                'power_play_time'])]
        return train_data

    def hyper_parameter_tuning(self, train_data, train_label, param_grid):
        xgb_model = xgb.XGBClassifier()
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(train_data, train_label)
        return grid_search.best_params_, grid_search.best_score_

    def train_model(self, train_data, train_label, params):
        cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=101)
        folds = [(train, val)
                 for train, val in cv.split(train_data, train_label)]
        metrics = ['auc', 'fpr', 'tpr',
                   'thresholds', 'probabilities', 'labels', 'accuracy', 'confusion_matrix']
        results = {
            'train': {m: [] for m in metrics},
            'val': {m: [] for m in metrics},
        }
        for train, val in tqdm(folds, total=len(folds)):
            dtrain = xgb.DMatrix(
                train_data.iloc[train, :], label=train_label.iloc[train])
            dval = xgb.DMatrix(
                train_data.iloc[val, :], label=train_label.iloc[val])
            model = xgb.train(
                dtrain=dtrain,
                params=params,
                evals=[(dtrain, 'train'), (dval, 'val')],
                num_boost_round=1000,
                verbose_eval=False,
                early_stopping_rounds=10,
            )
            sets = [dtrain, dval]
            for i, ds in enumerate(results.keys()):
                y_preds = model.predict(sets[i])
                labels = sets[i].get_label()
                fpr, tpr, thresholds = roc_curve(labels, y_preds)
                results[ds]['fpr'].append(fpr)
                results[ds]['tpr'].append(tpr)
                results[ds]['thresholds'].append(thresholds)
                results[ds]['auc'].append(roc_auc_score(labels, y_preds))
                results[ds]['probabilities'].append(y_preds)
                results[ds]['labels'].append(labels)
                predictions = y_preds > 0.5
                predictions = predictions.astype(int)
                results[ds]['accuracy'].append(
                    accuracy_score(labels, predictions))
                results[ds]['confusion_matrix'].append(
                    confusion_matrix(labels, predictions))

        return results

    def plot_roc(self, results, save_path):
        kind = 'val'
        c_fill = 'rgba(52, 152, 219, 0.2)'
        c_line = 'rgba(52, 152, 219, 0.5)'
        c_line_main = 'rgba(41, 128, 185, 1.0)'
        c_grid = 'rgba(189, 195, 199, 0.5)'
        c_annot = 'rgba(149, 165, 166, 0.5)'
        c_highlight = 'rgba(192, 57, 43, 1.0)'
        fpr_mean = np.linspace(0, 1, 10)
        interp_tprs = []
        for i in range(10):
            fpr = results[kind]['fpr'][i]
            tpr = results[kind]['tpr'][i]
            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std = 2*np.std(interp_tprs, axis=0)
        tpr_upper = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower = tpr_mean-tpr_std
        auc = np.mean(results[kind]['auc'])
        fig = go.Figure([
            go.Scatter(
                x=fpr_mean,
                y=tpr_upper,
                line=dict(color=c_line, width=1),
                hoverinfo="skip",
                showlegend=False,
                name='upper'),
            go.Scatter(
                x=fpr_mean,
                y=tpr_lower,
                fill='tonexty',
                fillcolor=c_fill,
                line=dict(color=c_line, width=1),
                hoverinfo="skip",
                showlegend=False,
                name='lower'),
            go.Scatter(
                x=fpr_mean,
                y=tpr_mean,
                line=dict(color=c_line_main, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f'AUC: {auc:.3f}')
        ])
        fig.add_shape(
            type='line',
            line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            title='ROC curve and AUC metric',
            template='plotly_white',
            title_x=0.5,
            xaxis_title="1 - Specificity",
            yaxis_title="Sensitivity",
            width=800,
            height=800,
            legend=dict(
                yanchor="bottom",
                xanchor="right",
                x=0.95,
                y=0.01,
            )
        )
        fig.update_yaxes(
            range=[0, 1],
            gridcolor=c_grid,
            scaleanchor="x",
            scaleratio=1,
            linecolor='black')
        fig.update_xaxes(
            range=[0, 1],
            gridcolor=c_grid,
            constrain='domain',
            linecolor='black')

        fig.show()
        fig.write_image(save_path)

    def plot_roc_simple(self, models, results, save_path):
        for j, model in enumerate(models):
            fpr_mean = np.linspace(0, 1, 10)
            interp_tprs = []
            for i in range(10):
                fpr = results[j]['val']['fpr'][i]
                tpr = results[j]['val']['tpr'][i]
                interp_tpr = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean = np.mean(interp_tprs, axis=0)
            auc = np.mean(results[j]['val']['auc'])
            plt.plot(fpr_mean, tpr_mean,
                     label=f"{model} (area = {round(auc, 2)})")

        plt.plot(fpr_mean, fpr_mean, linestyle='--',
                 color='r', label='random guess')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.show()

    def plot_goal_rate(self, models, results, save_path):
        for j, model in enumerate(models):
            goal_rate_mean = []
            for i in range(len(results[j]['val']['probabilities'])):
                df = pd.DataFrame(
                    {'isgoal': results[j]['val']['labels'][i], 'probabilities': results[j]['val']['probabilities'][i]})
                # calculate shot probability model percentile for each shot
                df['percentile'] = df['probabilities'].rank(pct=True)
                # group shots by percentile and calculate goal rate for each group
                goal_rates = df.groupby(pd.cut(df['percentile'], bins=[
                                        0]+[i/10 for i in range(1, 11)], include_lowest=True), observed=False)['isgoal'].mean()
                goal_rate_mean.append(goal_rates)

            goal_rate_mean = np.array(goal_rate_mean).mean(axis=0)
            goal_rates_mid = []
            for i in range(len(goal_rates.index)):
                goal_rates_mid.append(goal_rates.index[i].mid)

            # plot the goal rates as a function of the shot probability model percentile
            plt.plot(np.array(goal_rates_mid) * 100,
                     goal_rate_mean, label=model)

        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(np.arange(0, 110, 10))
        plt.gca().invert_xaxis()  # invert the x-axis
        plt.xlabel('Shot Probability Model Percentile')
        plt.title('Goal Rate')
        plt.ylabel('Goals / (Shots + Goals)')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.show()

    def plot_cumulative(self, models, results, save_path):
        for j, model in enumerate(models):
            goal_rate_mean = []
            for i in range(len(results[j]['val']['probabilities'])):
                df = pd.DataFrame(
                    {'isgoal': results[j]['val']['labels'][i], 'probabilities': results[j]['val']['probabilities'][i]})
                # calculate shot probability model percentile for each shot
                df['percentile'] = df['probabilities'].rank(pct=True)
                # group shots by percentile and calculate goal rate for each group
                goal_rates = df.groupby(pd.cut(df['percentile'], bins=[
                                        0]+[i/10 for i in range(1, 11)], include_lowest=True), observed=False)['isgoal'].mean()
                # Calculate the frequency and percentage of goals for each group
                goal_freq = goal_rates * goal_rates.index.to_series().apply(lambda x: x.length)
                goal_perc = goal_freq / goal_freq.sum()
                # cum_goal_perc = goal_perc.cumsum(axis=0)
                cum_goal_perc = np.cumsum(goal_perc[::-1])[::-1]
                goal_rate_mean.append(cum_goal_perc)

            goal_rate_mean = np.array(goal_rate_mean).mean(axis=0)
            goal_rates_mid = []
            for i in range(len(goal_rates.index)):
                goal_rates_mid.append(goal_rates.index[i].mid)

            goal_rate_mean = np.append(goal_rate_mean, 0)
            goal_rates_mid = np.append(goal_rates_mid, 1)

            # plot the goal rates as a function of the shot probability model percentile
            plt.plot(np.array(goal_rates_mid) * 100,
                     goal_rate_mean, label=model)

        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(np.arange(0, 110, 10))
        plt.gca().invert_xaxis()  # invert the x-axis
        plt.xlabel('Shot Probability Model Percentile')
        plt.ylabel('Proportion')
        plt.title('Cumulative % of Goals')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()

    def plot_calibration_curve(self, models, results, save_path):
        fig = plt.figure()
        plt.title(f"Calibration Curve")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for i in range(len(models)):
            y_prob = functools.reduce(
                operator.iconcat, results[i]['val']['probabilities'], [])
            y_true = functools.reduce(
                operator.iconcat, results[i]['val']['labels'], [])
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=20)
            plt.plot(prob_pred, prob_true, "s-", label="%s" % (models[i]))

        plt.xlabel("P(Pred)")
        plt.ylabel("P(Goal|Pred)")
        plt.legend()

        plt.xticks(np.arange(0, 1.2, 0.2))
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.show()

        fig.savefig(save_path)

    def log_experiment(self, results, experiment_tag, log_model, model_name, model_path, hp, hparameters):
        experiment = Experiment(api_key=os.getenv("COMET_API_KEY"), project_name=os.getenv(
            "COMET_PROJECT_NAME"), workspace=os.getenv("COMET_WORKSPACE"))
        accuracy = np.mean(results['val']['accuracy'])
        roc = np.mean(results['val']['auc'])
        confusion_matrix = np.mean(
            results['val']['confusion_matrix'], dtype=np.int32, axis=0)
        metrics = {"acc": accuracy, 'roc': roc}
        experiment.log_metrics(metrics)
        experiment.add_tag(experiment_tag)
        experiment.log_confusion_matrix(matrix=confusion_matrix)
        if hp:
            experiment.log_parameters(hparameters)
        if log_model:
            experiment.log_model(model_name, model_path)

        experiment.end()

    def train_model_nocv(self, train_data, train_label, params):
        dtrain = xgb.DMatrix(train_data, label=train_label)

        estop = 10
        n_folds = 5
        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            nfold=n_folds,
            metrics='logloss',
            early_stopping_rounds=estop,
            stratified=True,
            seed=42
        )

        # ref: https://stackoverflow.com/questions/40500638/xgboost-cv-and-best-iteration
        best_nrounds = int((cv_results.shape[0] - estop) / (1 - 1 / n_folds))
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=best_nrounds,
            verbose_eval=False
        )

        return model
