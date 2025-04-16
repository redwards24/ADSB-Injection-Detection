
from typing import List, Literal, Tuple
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint, loguniform
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, loguniform, uniform
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

import os
os.system('cls')

random_state = 345


class Metrics():

    def __init__(self):
        self.t_time: list = []
        self.v_time: list = []
        self.acc: list = []
        self.pr: list = []
        self.rc: list = []
        self.f1: list = []
        self.cm: list = []

    def accumulate(self, t_time, v_time, y_test, y_pred):
        self.t_time.append(t_time)
        self.v_time.append(v_time)
        self.acc.append(metrics.accuracy_score(y_test, y_pred))
        self.pr.append(metrics.precision_score(y_test, y_pred, average='weighted'))
        self.rc.append(metrics.recall_score(y_test, y_pred, average='weighted'))
        self.f1.append(metrics.f1_score(y_test, y_pred, average='weighted'))
        self.cm.append(confusion_matrix(y_test, y_pred))

    def avg(self, metric: list):
        return sum(metric) / len(metric)

    def evaluate(self):
        self.avg_acc = self.avg(self.acc) * 100
        self.avg_pr = self.avg(self.pr)
        self.avg_rc = self.avg(self.rc)
        self.avg_f1 = self.avg(self.f1)
        self.avg_t_time = self.avg(self.t_time) * 1000
        self.avg_v_time = self.avg(self.v_time) * 1000
        self.avg_cm = np.mean(np.array(self.cm), axis=0)

        self.tp = np.diag(self.avg_cm)
        self.fp = self.avg_cm.sum(axis=0) - self.tp
        self.fn = self.avg_cm.sum(axis=1) - self.tp
        self.tn = self.avg_cm.sum() - (self.tp + self.fp + self.fn)

        self.avg_cls_dr = self.tp / (self.tp + self.fn) * 100
        self.avg_cls_mdr = self.fn / (self.tp + self.fn) * 100
        self.avg_cls_far = self.fp / (self.fp + self.tn) * 100

#
# Define Evaluation Function for Estimator
#
def evaluate(
    estimator: BaseEstimator, 
    X_train, X_test,
    y_train, y_test,
    cv=10
):

    metrics = Metrics()

    for _ in range(cv):
        t = time.time()
        estimator.fit(X_train, y_train)
        t_time = time.time() - t

        t = time.time()
        y_pred = estimator.predict(X_test)
        v_time = time.time() - t

        metrics.accumulate(
            t_time=t_time, 
            v_time=v_time,
            y_test=y_test,
            y_pred=y_pred
        )

    metrics.evaluate()
    return estimator, metrics




#
# Define Random Search Function
#
def random_search(
        estimator: BaseEstimator, 
        params: dict,
        cv: int = 10,
        n_iter: int = 100,
        n_jobs: int = -1,
        verbose: int = 0
) -> Tuple[RandomizedSearchCV, float]:
    
    """
    Perform Random Search on the given estimator and parameter distribution.\n
    Returns the fitted random search object along with the search time.
    """

    # Create random search object
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    start = time.time()             # Start Time
    search.fit(X_train, y_train)    # Perform Random Search
    stop = time.time()              # Stop Time

    return search, (stop - start)

    # search_time = (stop - start) / 60
    # best_params = json.dumps(search.best_params_, indent=4)

    # print("\nTotal Random Search Time(min): ", search_time)
    # print("Best Params: ", best_params)

    # with open(params_output_file, 'a') as file:
    #     file.write(f"{name}: {search_time},{best_params}\n")

    # evaluate(estimator=search.best_estimator_, name=name)

    # return search


#
# Define Model Keys
#

RF: str = "RF"
KNN: str = "KNN"
MLP: str = "MLP"
LR: str = "LR"
DT: str = "DT"
NB: str = "NB"

estimator_dict = {
    RF: RandomForestClassifier,
    KNN: KNeighborsClassifier,
    MLP: MLPClassifier,
    LR: LogisticRegression,
    DT: DecisionTreeClassifier,
    NB: GaussianNB
}

params_dict = {
    RF: {
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'n_estimators': randint(100, 1000),
        'max_depth': randint(10, 200),
        'min_samples_split': randint(2, 50), 
        'min_samples_leaf': randint(1, 50), 
        'ccp_alpha': loguniform(1e-4, 1e-1),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    KNN: {
        'n_neighbors': randint(1, 30),
        'leaf_size': randint(10, 100), 
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], 
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
        'p': randint(1, 5)
    },
    MLP: {
        'hidden_layer_sizes': [(100,), (100, 50), (200,), (150, 100, 50), (300, 200)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'alpha':loguniform(0.00001,2),
        'max_iter': randint(200,1000),
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': loguniform(1e-4, 1e-1),
        'momentum': uniform(0.0, 0.9),
        'early_stopping': [True, False]
    },
    LR: {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'], 
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': randint(100,1000), 
        'C': loguniform(1e-4,1e4),
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
    },
    DT: {
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'splitter': ['best', 'random'], 
        'max_depth': randint(10, 200), 
        'min_samples_split': randint(2, 50),
        'min_samples_leaf': randint(1, 50), 
        'ccp_alpha': loguniform(1e-4, 1e-1),
        'max_features': ['sqrt', 'log2', None]
    },
    NB: {
        'var_smoothing': loguniform(1e-9, 1e-6)
    }
}



def feature_correlation(
        data: pd.DataFrame,
        save_path: str = "FeatureCorrelation.png",
        method: Literal['pearson', 'kendall', 'spearman'] = "spearman",
        create_fig: bool = False,
        show_plot: bool = False
) -> pd.DataFrame:
    """
    Perform Feature Correlation on the given dataset and saves the result as a png in the given path.\n
    Returns the correlation matrix as a pandas DataFrame.
    """
    correlation_matrix = data.corr(method=method)
    if create_fig:
        plt.figure(figsize=(56, 25))
        sns.set(font_scale=4.5)
        sns.heatmap(correlation_matrix, 
                    linewidths=4, 
                    annot_kws={"fontsize": 38, "fontweight": "bold"}, 
                    annot=True, 
                    fmt=".2f",
                    cmap='Reds',
                    vmin=-1, vmax=1)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        if show_plot:
            plt.show()
    return correlation_matrix



def feature_importance(
        X, y,
        estimator: BaseEstimator = RandomForestClassifier(n_estimators=560, max_depth=177, random_state=random_state),
        display_names: List[str] = ['lat', 'lon', 'vel', 'heading', 'vertrate', 'baroalt', 'geoalt'],
        save_path: str = "FeatureImportance.png",
        create_fig: bool = False,
        show_plot: bool = False
) -> pd.Series:
    """
    Fits the given estimator on the given data and extracts the feature importances.\n
    The importances are plotted on a bar chart and the plot is saved to the given path.\n
    Returns the importances as a pandas Series.
    """
    estimator.fit(X, y)
    mdi_importances: pd.Series = pd.Series(estimator.feature_importances_, index=display_names).sort_values(ascending=True)
    mdi_importances = (mdi_importances / mdi_importances.max()) * 100
    if create_fig:
        font = {
            'family': 'sans-serif',
            'weight': 'normal',
            'size': 18
        }
        plt.figure(figsize=(56, 25))
        matplotlib.rc('font', **font)
        ax = mdi_importances.plot.barh()
        ax.set_xlabel("Relative importance (%)")
        ax.set_ylabel("Feature")
        ax.set_xlim(0, 100)
        ax.figure.tight_layout()
        plt.savefig(save_path, dpi=600)
        if show_plot:
            plt.show()
    return mdi_importances



def load_data(
        csv_path: str,
        x_cols: List[int] = [0, 1, 2, 3, 4, 6, 7],
        y_cols: List[int] = [8]
) -> List[pd.DataFrame]:
    """
    Loads the data from the given csv_path.
    """
    data = pd.read_csv(csv_path)
    X = data.iloc[:, x_cols]
    y = data.iloc[:, y_cols]
    return X, y


if __name__ == "__main__":

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Change prefix to OHARE for ohare dataset
    # Change prefix to 10_AIRPORT for 10 airport dataset
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    prefix = "OHARE"

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Change based on OHARE dataset or 10 airport dataset
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    input_file = "OHARE_authentic_attack.csv"

    feature_correlation_save_path = prefix + "_feature_correlation.png"
    feature_importance_save_path = prefix + "_feature_importance.png"

    params_output_file = prefix + "_random_search_results.txt"
    metrics_output_file = prefix + "_metrics.txt"

    # Load X and y data sets
    print("Loading dataset...")
    X, y = load_data(input_file)
    col_names = X.columns
    y = y.to_numpy().ravel()

    # Create Scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Perform Feature Correlation
    print("\nPerforming Feature Correlation...")
    corr: pd.DataFrame = feature_correlation(
        data=X,
        save_path=feature_correlation_save_path,
        create_fig=True,
    )

    # Find Highly Correlated Indicies
    threshold = 0.8
    corr = corr.gt(threshold)

    # Ignore Diagonal (lat correlated with lat is 1)
    for i in range(len(corr)):
        corr.iloc[i, i] = False

    # Check if any features are correlated
    if corr.any().any(): # check over rows and cols

        print("Found Correlated Features:")
        # Find correlated columns
        corr_cols = []
        for item in corr.items():                       # for each entry
            for key in item[1].index:                   # for each feature
                if item[1][key]:                        # if feature is correlated
                    new_pair = sorted((item[0], key))   # create new correlated pair sorted alphabetically
                    if new_pair not in corr_cols:       # if pair is unique
                        corr_cols.append(new_pair)      # add to correlated columns
        for pair in corr_cols:
            print(pair)


        # There are correlated features
        # -> Perform feature importance
        print("\nPerforming Feature importance...")
        importances: pd.Series = feature_importance(
            scaler.transform(X), y, 
            display_names=col_names, 
            save_path=feature_importance_save_path,
            create_fig=True,
            estimator=DecisionTreeClassifier()
        )

        # Find Least important features
        dropped_features = []
        for pair in corr_cols:
            if importances[pair[0]] > importances[pair[1]]:
                dropped_features.append(pair[1])
            else:
                dropped_features.append(pair[0])
        print("Dropping features:", dropped_features)

        # Remove least important features
        X = X.drop(dropped_features, axis=1)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(scaler.fit_transform(X), y, train_size=0.7, random_state=random_state)

    # Perform Search and Evaluation on each model    
    for model in [RF, KNN, MLP, LR, DT, NB]:

        print(f"\n\nModel: {model}")

        print(f"\nPerfoming Random Search...")
        search, search_time = random_search(
            estimator=estimator_dict[model](),
            params=params_dict[model]
        )
        print(f"Random Search Finished.\tTotal Seach Time: {search_time / 60} min.")

        print(f"\nEvaluating Best Model...")
        fitted_model, model_metrics = evaluate(
            estimator=estimator_dict[model](**search.best_params_),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        print(f"Evaluation Finished.")
        
        print(f'\nResults for {model}:')
        print(f'Accuracy: {model_metrics.avg_acc}')
        print(f'Precision: {model_metrics.avg_pr}')
        print(f'Recall: {model_metrics.avg_rc}')
        print(f'F1: {model_metrics.avg_f1}')
        print(f'DR: {model_metrics.avg_cls_dr}')
        print(f'MDR: {model_metrics.avg_cls_mdr}')
        print(f'FAR: {model_metrics.avg_cls_far}')
        print(f'Training Time(ms): {model_metrics.avg_t_time}')
        print(f'Validation Time(ms): {model_metrics.avg_v_time}')

        print("\nSaving results...")
        
        # Save Best Hyperparameters
        with open(params_output_file, 'a') as file:
            file.write(f"{model}: {search_time},{json.dumps(search.best_params_, indent=4)}\n")

        # Save Metrics
        with open(metrics_output_file, 'a') as file:
            file.write(
                f'{model}: ' +
                f'{model_metrics.avg_acc:.2f},' +
                f'{model_metrics.avg_pr:.2f},' +
                f'{model_metrics.avg_rc:.2f},' +
                f'{model_metrics.avg_f1:.2f},' +
                f'{model_metrics.avg_cls_dr[0]:.2f},' +
                f'{model_metrics.avg_cls_dr[1]:.2f},' +
                f'{model_metrics.avg_cls_dr[2]:.2f},' +
                f'{model_metrics.avg_cls_dr[3]:.2f},' +
                f'{model_metrics.avg_cls_mdr[0]:.2f},' +
                f'{model_metrics.avg_cls_mdr[1]:.2f},' +
                f'{model_metrics.avg_cls_mdr[2]:.2f},' +
                f'{model_metrics.avg_cls_mdr[3]:.2f},' +
                f'{model_metrics.avg_cls_far[0]:.2f},' +
                f'{model_metrics.avg_cls_far[1]:.2f},' +
                f'{model_metrics.avg_cls_far[2]:.2f},' +
                f'{model_metrics.avg_cls_far[3]:.2f},' +
                f'{model_metrics.avg_t_time:.2f},' +
                f'{model_metrics.avg_v_time:.2f}\n'
            )
        
        # Save Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=model_metrics.cm[0], display_labels=['Auth', 'PM', 'GI', 'VD'])
        font = {'weight' : 'normal', 'size'   : 18}
        matplotlib.rc('font', **font)
        disp.plot(cmap=plt.cm.GnBu, colorbar=False)
        plt.savefig(prefix + "_" + model + "_cm", dpi=600, bbox_inches='tight')

        # Save Model
        import pickle
        with open(f'{prefix}_{model}_model.pkl', 'wb') as file:
            pickle.dump(fitted_model, file)

        print("Saved.")








