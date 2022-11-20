# Essential libraries
import numpy as np
import pandas as pd
from functools import partial
import itertools
from sklearn.base import clone
from sklearn.experimental import enable_hist_gradient_boosting

# For plotting and display
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
from IPython.display import HTML, display
from tqdm.autonotebook import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
from math import ceil
import matplotlib

# Set negative lines as solid
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# For feature sampling
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# For scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# For scoring metrics
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import hinge_loss, matthews_corrcoef, roc_auc_score
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, log_loss
from sklearn.metrics import precision_score, recall_score, zero_one_loss
from sklearn.metrics import explained_variance_score, max_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error
from sklearn.metrics import make_scorer

# For inspection
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance

# For ML Models
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.linear_model import SGDClassifier, SGDRegressor, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

# XGBoost, Catboost, and LightGBM requires library installation
try:
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from catboost import CatBoostClassifier, CatBoostRegressor
    adv_gbm = True
except:
    adv_gbm = False

# Define function to create percentage tickers
def percent(x, pos):
    """Create percentage format for tickers"""
    return "{:.1f}%".format(x*100)

class MLModels:
    # safe to change
    random_state = None # Set random state / discarded by leo
    reproducible = True # Set to true if reproducible run is desired

    # not so safe to change
    model = None
    _setting_name = None

    def __init__(self):
        self.training_accuracy = None
        self.test_accuracy = None
        self.training_std = None
        self.test_std = None
        self.coef = None
        self.classes = None
        self._setting = None

    def list_all_methods():
        """Print a list of all possible methods"""
        # Print header
        print("Classification\n"+"-"*14)

        # Print classification methods
        i = 0
        for m in MLModels.all_methods()['Classification'].keys():
            print(f"{i+1}. {m}")
            i += 1

        # Print Regression header
        print("\nRegression\n"+"-"*10)

        # Print regression methods
        for i, m in enumerate(MLModels.all_methods()['Regression'].keys()):
            print(f"{i+1}. {m}")

    def all_methods(
            n_nb=list(range(1, 51)),
            C=[1e-5, 1e-4, 1e-3, .01, 0.1, 0.2, 0.4, 0.75, 1, 1.5, 3, 5, 10,
               15, 20, 100, 1000, 5000, 10000, 50000, 100000],
            max_depth=list(range(1, 51)),
            tree_rs=None):
        """
        Return a dict of list of classification and regression methods

        Returns a list of all available methods initialized with their
        corresponding parameters. This function was defined so that the user
        may easily see what machine learning methods are available, and for
        the coder to easily integrate the new machine learning methods.

        Parameters
        ----------
        n_nb : list of int
            Number of nearest neighbors setting
        C : list of float
            Regularization constant setting
        max_depth : list of int
            Max depth setting for trees
        tree_rs : in
            Random state setting for trees

        Returns
        -------
        methods : dict
            Dictionary containing the initialized MLModels objects
        """
        # Initialize methods container
        methods = {}

        # Set classification methods
        methods['Classification'] = {
            'kNN': KNNClassifier(n_nb),
            'Logistic (L1)': LogisticRegressor(C, 'l1'),
            'Logistic (L2)': LogisticRegressor(C, 'l2'),
            'SG Logistic (L1)': SGDC(C, 'l1', 'log'),
            'SG Logistic (L2)': SGDC(C, 'l2', 'log'),
            'SVM (L1)': LinearSVM(C, 'l1'),
            'SVM (L2)': LinearSVM(C, 'l2'),
            'SVM RBF': NonLinearSVM(C, 'rbf'),
            'SG SVM (L1)': SGDC(C, 'l1', 'hinge'),
            'SG SVM (L2)': SGDC(C, 'l2', 'hinge'),
            'Decision Tree': DecisionTree(max_depth, tree_rs),
            'RF Classifier': RFClassifier(max_depth, tree_rs),
            'ET Classifier': ETClassifier(max_depth, tree_rs),
            'GB Classifier': GBClassifier(max_depth, tree_rs),
            'Histogram GBC': HistGBC(max_depth, tree_rs),
            'AdaBoost DT': ABDTClassifier(max_depth, tree_rs),
            'Multinomial NB': MNB(C),
            'Complement NB': CNB(C),
            'Bernoulli NB': BNB(C),
            'MLP Classifier': MLPC(C)}
        if adv_gbm:
            methods['Classification'].update({
                'XGB Classifier': XGBC(max_depth, tree_rs),
                'LightGBM Classifier': LGBMC(max_depth, tree_rs),
                'CatBoost Classifier': CBClassifier(max_depth, tree_rs),})

        # Set Regression methods
        methods['Regression'] = {
            'kNN': KNNRegressor(n_nb),
            'Ridge': RidgeRegressor(C),
            'Lasso': LassoRegressor(C),
            'Elastic Net': EN(C),
            'SVR': SVMRegressor(C),
            'SVR (RBF)': SVMRegressor(C, 'rbf'),
            'SG SVR': SGDR(C, 'l2', 'epsilon_insensitive'),
            'SGD Regressor (L2)': SGDR(C, 'l2', 'squared_loss'),
            'SGD Regressor (L1)': SGDR(C, 'l1', 'squared_loss'),
            'GP Regressor': GPRegressor(C),
            'Decision Tree': DTRegressor(max_depth, tree_rs),
            'RF Regressor': RFRegressor(max_depth, tree_rs),
            'ET Regressor': ETRegressor(max_depth, tree_rs),
            'GB Regressor': GBRegressor(max_depth, tree_rs),
            'Histogram GBR': HistGBR(max_depth, tree_rs),
            'AdaBoost DT': ABDTRegressor(max_depth, tree_rs),
            'MLP Regressor': MLPR(C)}
        if adv_gbm:
            methods['Regression'].update({
                'XGB Regressor': XGBR(max_depth, tree_rs),
                'LightGBM Regressor': LGBMR(max_depth, tree_rs),
                'CatBoost Regressor': CBRegressor(max_depth, tree_rs),})

        return methods

    def scalers():
        """Return a dict of all scaler methods"""
        return {'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'maxabs': MaxAbsScaler(),
                'robust': RobustScaler(),
                'power': PowerTransformer(),
                'quantile': QuantileTransformer()}

    def scorers():
        """Return a dict of all scorers"""
        return {'balanced': balanced_accuracy_score,
                'cohen_kappa': cohen_kappa_score,
                'hinge': hinge_loss,
                'matthews': matthews_corrcoef,
                'roc_auc': roc_auc_score,
                'f1': f1_score,
                'hamming': hamming_loss,
                'jaccard': jaccard_score,
                'log_loss': log_loss,
                'precision': partial(precision_score, zero_division=0),
                'recall': recall_score,
                'zero_one': zero_one_loss,
                'explained_variance': explained_variance_score,
                'max_error': max_error,
                'mean_absolute': mean_absolute_error,
                'mean_squared': mean_squared_error,
                'mean_squared_log': mean_squared_log_error,
                'median_absolute': median_absolute_error}

    def plot_results(methods):
        """
        Plot the accuracies of all the trained machine learning methods given

        Parameters
        ----------
        methods : dict
            Dictionary containing the trained machine learning models

        Returns
        -------
        axes : list of matplotlib axes
            Axes of all the trained methods
        """
        # Initialize axes container
        axes = []

        # Iterate through all methods
        for k in methods:
            # Get trained machine learning method
            m = methods[k]

            # Plot the results of the current method
            ax = m.plot_accuracy()

            # Set title
            ax.set_title(k, fontsize=16)

            # Show figure
            plt.show()

            # Append axes
            axes.append(ax)

        return axes

    def plot_accuracy(self):
        """
        Plots the train and test accuracy of the model for various settings

        Returns
        -------
        ax : matplotlib.Axes
            Axes of the figure plotted
        """
        # Initialize figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Plot train and test accuracy points
        ax.plot(self._setting, self.training_accuracy,
                label="training accuracy", lw=3.0, color='tab:blue')
        ax.plot(self._setting, self.test_accuracy,
                label="test accuracy", lw=3.0, color='tab:orange')

        # Visualize the standard deviation as a filled area
        ax.fill_between(self._setting,
                        self.training_accuracy - self.training_std,
                        self.training_accuracy + self.training_std, alpha=0.2,
                        color='tab:blue')
        ax.fill_between(self._setting, self.test_accuracy - self.test_std,
                        self.test_accuracy + self.test_std, alpha=0.2,
                        color='tab:orange')

        # Set axis labels
        ax.set_ylabel("Accuracy", fontsize=14)

        # Check if n_neighbors
        if self._setting_name == 'n_neighbors':
            ax.set_xlabel("$n_{neighbors}$", fontsize=14)
        elif self._setting_name == 'max_depth':
            ax.set_xlabel("Max Depth", fontsize=14)
        else:
            ax.set_xlabel("${}$".format(self._setting_label), fontsize=14)

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set axis scale depending on method
        if self.log_plot:
            ax.set_xscale('log')

        # Set y axis tickers as percentages
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(percent))

        # Get xlim and ylim
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]

        # Draw legends
        ax.text(s="Train", fontsize=14, weight='bold', color='tab:blue',
                x=np.array(self._setting).max() + xrange*0.01,
                y=self.training_accuracy[-1] + yrange*0.015)
        ax.text(s="Test", fontsize=14, weight='bold', color='tab:orange',
                x=np.array(self._setting).max() + xrange*0.01,
                y=self.test_accuracy[-1] - yrange*0.015)

        return ax

    def plot_feature_importance(self, feature_names=None, num_feat=10):
        """
        Plots the imporatance of each feature in descending order

        Parameters
        ----------
        self : MLModels object
            Trained decision tree type model
        feature_names : list-like
            String of list to be used as feature names
        num_feat : int
            Top `n_feat` to be shown in the plot

        Returns
        -------
        ax : matplotlib axes
            Matplotlib axes of the feature importance plot
 
        Examples
        --------
        >>> m = MLModels.run_classifier(X, label)
        >>> ax = m['Decision Tree'].plot_feature_importance(feature_names)
        """
        # Set num feat depending on the number of features
        num_feat = min([len(self.feature_importance), num_feat])

        # Get weights and indices
        weights = self.feature_importance
        indices = weights.argsort()[::-1][:num_feat]

        # Initialize figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Plot feature importances
        if feature_names is not None:
            ax.barh(np.array(feature_names)[indices][::-1],
                    weights[indices][::-1])
        else:
            ax.barh(range(num_feat), weights[indices][::-1])

        # Set axis labels
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return ax

    def plot_coef(self, feature_names=None, cmap='plasma'):
        """
        Plots the coefficient of each feature for linear methods per parameter

        Returns
        -------
        ax : matplotlib axes
            Matplotlib axes of the plot

        Examples
        --------
        >>> m = MLModels.run_classifier(data, target,
                            use_methods=['SVM (L2)'],
                            max_depth=range(1, 10), n_trials=10)
        >>> ax = m['SVM (L2)'].plot_coef()
        """
        # Initialize figure
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot(111)

        # Get max and minimum values
        max_coef = np.max(self.all_coef)
        min_coef = np.min(self.all_coef)

        # Reverse the plot order for C regressions
        if self._setting_name == 'C':
            to_plot = self.all_coef[::-1]
        else:

            to_plot = self.all_coef

        # Set color cycle
        colors = sns.color_palette(cmap, len(self.all_coef))

        # Iterate through all coefficients
        for i, coefs in enumerate(to_plot):
            # Plot coefficients in the graph
            ax.plot(np.arange(len(coefs)), coefs, '-o', color=colors[i],
                    lw=3.0, markersize=7.0, alpha=0.80)

        # Adjust y lim depending on min and max values
        if np.sign(max_coef*min_coef) > 0:
            ylim = max_coef*1.01
        else:
            ylim = min(np.abs([min_coef, max_coef]))*1.05
        ax.set_ylim(sorted([ylim, -ylim]))

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set axis labels
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_ylabel('Coefficient', fontsize=14)

        plt.xticks(range(len(coefs)))

        # Label ticks as features if feature names are given
        if feature_names is not None:
            ax.set_xticklabels(feature_names)

        return ax

    def plot_regularization(self, feature_names=None, num_settings=10,
                            start_num=0):
        """
        Plot the effect of regularization with respect to predictability

        Mainly intented for L1 and L2 regularization model. Generates several
        plots of the the features vs weights in descending order of
        importance. `num_settings` controls the number of plots to generate.
        While plots are ordered according to decreasing regularization.
        `start_num` may be specified in order to set the starting setting to
        be plotted.

        Parameters
        ----------
        self : MLModels object
            Trained L1 or L2 regularization model
        feature_names : list of str, default=None
            List of feature names to be used as labels in plotting
        num_setting : int, default=10
            Number of regularization plots to generate
        start_num : int, default=10
            Setting number to start the plots from

        Examples
        --------
        >>> m = MLModels.run_classifier(X, label, n_trials=2,
                            use_methods=['Logistic (L1)'])
        >>> m['Logistic (L1)'].plot_regularization(feature_names)
        """
        # Set last num
        last_num = min(len(self._setting), start_num + num_settings)

        # Set range depending if alpha or C setting
        if self._setting_name == 'alpha':
            setting_range = list(range(len(self._setting)))[::-1]
        else:
            setting_range = list(range(len(self._setting)))

        # Iterate through all settings
        for setting_num in setting_range[start_num:last_num]:
            # Get weights and indices
            weights = np.abs(self.all_coef[setting_num])
            indices = (np.abs(self.all_coef[setting_num])
                         .argsort()[::-1][:10])

            # Initialize figure
            fig = plt.figure()
            ax = fig.add_subplot(111)

            if feature_names is None:
                feature_names = list(map(str, range(len(weights))))

            # Plot points
            ax.barh(np.array(feature_names)[indices][::-1],
                    weights[indices][::-1])

            # Plot accuracy using anchored text
            at = AnchoredText(
                    "Accuracy: {:.2f}%".format(
                            self.test_accuracy[setting_num]),
                    loc='lower right', prop=dict(fontsize=10), frameon=False)
            ax.add_artist(at)

            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Show figure
            plt.show()

    def get_included_features(self, setting, tol=1e-7):
        """Return the included features during regularization given setting"""
        # Get setting index
        setting_num = self._setting.index(setting)

        return self.feature_names[
            (np.abs(self.all_coef) > 1e-7)[setting_num, :]]

    def plot_accuracy_num_features(self, num_pts=None, tol=1e-7):
        """
        Plot the accuracy versus numbe of features used in L1/L2 reg

        Mainly intended for feature selection/feature importance plotting of
        L1, L2 or Elastic Net regularization. Generates a plot showing the
        accuracy of the model versus the number of features used by the model.

        Parameters
        ----------
        self : MLModels object
            Trained L1, L2, or ElasticNet model
        num_pts : int
            Number of regularization points to plot. Useful if the result is
            to be truncated as desired.
        tol : float
            Value of the float before treating it as practically zero.
        """
        # Get variables to be plotted
        num_features = (np.abs(self.all_coef) > 1e-7).sum(1)[:num_pts]
        test_acc = self.test_accuracy[:num_pts]
        test_std = self.test_std[:num_pts]

        # Initialize figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Plot points
        ax.fill_between(num_features, test_acc - test_std,
                        test_acc + test_std, alpha=0.2, color='tab:blue')
        ax.plot(num_features, test_acc, 'o-', lw=2.5, markersize=6,
                color='tab:blue')

        # Set axis formatter
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(percent))

        # Set axis labels
        ax.set_xlabel('Number of features', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)

        # Show figure
        plt.show()

    def train_test(self, X, y, n_trials=100, scorer=None,
                   stratify=False, smote=False, test_size=0.25,
                   scaling=None, class_weight=None):
        """
        Calculate the train and test accuracy given input and target features
        
        Parameters
        ----------
        self : MLModels Object
            The machine learning model object
        X : np.array
            Input features
        y : np.array
            Target features
        n_trials : int
            Number of trials to do per hyper parameter
        scorer : str, default=None
            Specifies which scorer to be used in training and testing.
            See MLModels.scorers().keys() for a full list of available scoring
            metrics.
        stratify : bool, default=False
            Specify whether to perform stratified splitting
        smote : bool, default=False
            Specify whether to perform SMOTE
        test_size : float, default=0.25
            Specify the size of test dataset
        scaling : str, default=None
            Specify the scaling to be performed. By default does not scale the
            data. See MLModels.scalers() for available options.
        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form
            `{class_label: weight}`
        """
        # Initialize results container
        train_accuracies = []
        test_accuracies = []
        all_coef = []
        feature_importances = []
        train_times = []
        
        # Set n_trials
        self.n_trials = n_trials
        
        # Initialize progress bar
        pb = tqdm(total=self.n_trials*len(self._setting), miniters=1)
        
        # Iterate trials
        for i in range(self.n_trials):
            # Set random state if reproducible
            if self.reproducible:
                self.random_state = i

            # Split the test and training dataset
            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=self.random_state, stratify=y,
                    test_size=test_size)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                        X, y, random_state=self.random_state, 
                        test_size=test_size)

            # Perform scaling
            if scaling is not None:
                # Check if power scaling
                if scaling == 'power':
                    # Check if strictly positive features
                    if (X_train <= 0).sum() == 0:
                        # Set scaler
                        scaler = MLModels.scalers()['power']

                        # Set method to box cox
                        scaler.set_params(method='box-cox')
                    else:
                        scaler = MLModels.scalers()[scaling]
                else:
                    # Set scaler based on user option
                    scaler = MLModels.scalers()[scaling]

                # Fit to train dataset
                scaler.fit(X_train)

                # Transform train and test dataset
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            # Perform smote
            if smote:
                smt = SMOTE(random_state=self.random_state)
                X_train, y_train = smt.fit_sample(X_train, y_train)
            
            # Set description of progress bar
            pb.set_description(f'Iter: {i + 1}')

            # Initialize train test accuracy and coef results container
            training_accuracy = []
            test_accuracy = []
            feature_coef = []
            feature_importance = []
            train_time = []

            # Iterate to all setting
            for j, s in enumerate(self._setting):
                # Get start time
                start_time = time.time()
                
                # Initialize model
                if self._setting_name is not None:
                    clf = self.model(**{self._setting_name: s})
                else:
                    clf = self.model

                # Set class weights
                if class_weight is not None:
                    # Get class parameters
                    params = clf.get_params()

                    # Check if classifier has class weights attribute
                    if 'class_weight' in params.keys():
                        # Set class weight
                        params['class_weight'] = class_weight
                        clf.set_params(**params)

                # Train model
                clf.fit(X_train, y_train)
                
                # Get train time
                train_time.append(time.time() - start_time)

                # Record train and test accuracy
                if scorer is not None:
                    training_accuracy.append(MLModels.scorers()[scorer](
                            y_train, clf.predict(X_train)))
                    test_accuracy.append(MLModels.scorers()[scorer](
                            y_test, clf.predict(X_test)))
                else:
                    training_accuracy.append(clf.score(X_train, y_train))
                    test_accuracy.append(clf.score(X_test, y_test))


                # Get coefficient for the specified setting
                if hasattr(clf, 'coef_'):
                    if len(clf.coef_.shape) > 1:
                        # Take the mean of per axis
                        feature_coef.append(clf.coef_.mean(axis=0))                        
                    else:
                        # Append as usual
                        feature_coef.append(clf.coef_)
                # Get feature importances for Decision trees
                elif hasattr(clf, 'feature_importances_'):
                    feature_importance.append(clf.feature_importances_)

                # Update progress bar
                pb.update(1)

            # Append to overall accuracies and coef result container 
            train_accuracies.append(training_accuracy)
            test_accuracies.append(test_accuracy)
            train_times.append(train_time)
            feature_importances.append(feature_importance)
            if feature_coef:
                all_coef.append(feature_coef)


        # Close progress bar
        del pb

        # Compute for mean and std of train and test accuracy
        self.training_accuracy = np.mean(train_accuracies, axis=0)
        self.test_accuracy = np.mean(test_accuracies, axis=0)
        self.train_time = np.mean(train_times, axis=0)
        self.training_std = np.std(train_accuracies, axis=0)
        self.test_std = np.std(test_accuracies, axis=0)
        if hasattr(clf, 'feature_importances_'):
            self.feature_importance = np.mean(
                    feature_importances, axis=0).mean(axis=0)

        # For class with coefficients
        if feature_coef:
            # Get mean value of coefficients
            self.coef = np.mean(np.abs(all_coef), axis=0).mean(axis=0)
            self.all_coef = np.mean(np.abs(all_coef), axis=0)

    def run_classifier(X, labels, feature_names=None, C=None, max_depth=None,
                       n_neighbors=None, use_methods=None, n_trials=100,
                       tree_rs=None, scorer=None, task='C',
                       stratify=False, smote=False, test_size=0.25,
                       scaling=None, class_weight=None):
        """
        Perform accuracy measurements for various models given dataset
        
        By default, runs all models listed in `MLModels.list_all_methods()`
        on the given dataset with input features `X` and target features
        `labels`. Returns a dictionary with machine learning model as keys and
        corresponding accuracies as stored in the MLModels object as values.

        Parameters
        ----------
        X : np.array
            Input features
        labels : np.array
            Target features
        feature_names : list of string
            List of feature names
        C : list-like
            Regression coefficients to be used in testing (for L1 and L2)
        max_depth : list-like
            Maximum depth settings to be used for Decision Trees Training
        n_neighbors : list-like
            Number of neighbor settings to be used in testing (for KNN)
        use_methods : list-like
            Specify machine learning methods to be used.
            Execute `MLModels.list_all_methods()` for available models.
        n_trials : int
            Number of trials to be made in the training and testing
        tree_rs : int
            Random state to be used in Decision Tree classifier
        scorer : str, default=None
            Specifies which scorer to be used in training and testing.
            See MLModels.scorers().keys() for a full list of available scoring
            metrics
        task : str, default='C'
            Specify 'C' or 'R' if the task is classification or regression.
        stratify : boolean, default=False
            Specify if stratified train test splitting is preferred
        smote : boolean, default=False
            Specify to TRUE if SMOTE to be applied
        test_size : float, default=0.25
            Amount of test size in proportion to the dataset in splitting
        scaling : str, default=None
            Specify the scaling to be performed. By default does not scale the
            data. See MLModels.scalers() for available options.
        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form
            `{class_label: weight}`

        Returns
        -------
        trained_methods : dict
            Dictionary containing the model names and their corresponding
            accuracies as stored in the MLModels object
        
        Examples
        --------
        Training and Testing all models:
        >>> m = MLModels.run_classifier(data, target, n_trials=100)
        >>> res = MLModels.summarize(m, feature_names, show_plot=True,
                                     show_top=True)

        Select specific models:
        >>> m = MLModels.run_classifier(data, target,
                n_trials=100, use_methods=['SVM (L1)', 'SVM (L2)'])
        >>> res = MLModels.summarize(m, feature_names, show_plot=True, 
                                     show_top=True)
        """
        # Initialize C values
        if C is None:
            C = [1e-5, 1e-4, 1e-3, .01, 0.1, 0.2, 0.4, 0.75, 1, 1.5, 3, 5, 10,
                 15, 20, 100, 1000, 5000, 10000, 50000, 100000]

        # Initialize nearest neighbors value
        if n_neighbors is None:
            n_nb = list(range(1, 51))
        else:
            n_nb = n_neighbors
            
        # Initialize max_depth values
        if max_depth is None:
            max_depth = list(range(1, 51))
        
        # Initialize trained methods container
        trained_methods = {}
        
        # Enumerate methods to be used
        if task == 'C':
            methods = MLModels.all_methods(n_nb=n_nb, C=C,
                                           max_depth=max_depth,
                                           tree_rs=tree_rs)['Classification']
        elif task == 'R':
            methods = MLModels.all_methods(n_nb=n_nb, C=C,
                                           max_depth=max_depth,
                                           tree_rs=tree_rs)['Regression']
        
        # Check use_methods parameter
        if use_methods is None:
            # Perform all methods if nothing is specified
            use_methods = methods
        
        # Catch warnings
        with warnings.catch_warnings():
            # Fiter convergence warnings
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            # Iterate all methods specified
            for k in use_methods:
                # Print current model to be trained
                print("Training and testing: {} model".format(k))
                
                # Get current method in iteration
                m = methods[k]
                
                # Train method
                m.train_test(X, labels, n_trials=n_trials, scorer=scorer,
                             stratify=stratify, smote=smote, 
                             test_size=test_size, scaling=scaling,
                             class_weight=class_weight)

                # Set feature names
                if feature_names is not None:
                    m.feature_names = np.array(feature_names)
                else:
                    m.feature_names = np.array(
                            list(map(str, range(X.shape[1]))))

                # Append result
                trained_methods[k] = m

        return trained_methods

    def summarize(methods, feature_names=None, show_plot=False, 
                  show_top=False):
        """
        Summarize results of the training and testing of ML methods
        
        Gets the trained and tested machine learning models as input then
        shows the maximum accuracy and optimal parameter for each machine
        learning method in the input. Plots and top predictor may optionally
        be shown as well.
        
        Parameters
        ----------
        methods : dict
            Machine Learning method name as key
            Trained and tested ML method object as value
        feature_names : list of str
            List of feature names in the data
        show_plot : boolean
            Set True if accuracy plots are to be shown
        show_top : boolean
            Set True if top predictor is to be shown
        """
        # Initialize results variables
        names = []
        accuracies = []
        parameters = []
        features = []
        train_accuracies = []
        training_times = []
        
        # Iterate through all methods used
        for k in methods:
            # Get current method
            m = methods[k]
            
            # Append method name
            names.append(k)
            
            # Get maximum test accuracy setting
            max_index = m.test_accuracy.argsort()[-1]
            
            # Append accuracy
            accuracies.append('{:.2f}%'.format(
                    np.round(np.max(m.test_accuracy)*100, 2)))
            train_accuracies.append('{:.2f}%'.format(
                    np.round(m.training_accuracy[max_index]*100, 2)))
            training_times.append('{:.2f} secs'.format(
                    m.train_time[max_index]))
            
            # Append max test accuracy
            if m._setting_name == 'max_depth':
                parameters.append('' 
                                  + f'{m._setting_label}' 
                                  + '= {}'.format(
                        m._setting[np.argmax(m.test_accuracy)]))
            else:
                parameters.append('${} = {}$'.format(
                        m._setting_label, 
                        m._setting[np.argmax(m.test_accuracy)]))

            # Check if there are coefficients
            if hasattr(m, 'coef') or hasattr(m, 'feature_importance'):
                if m.coef is not None:
                    # Get index of maximum coefficient
                    tp = np.argmax(
                         np.abs(m.coef.flatten()))

                    # Match with corresponding feature
                    if feature_names is not None:
                        features.append(f'{feature_names[tp]}')
                    else:
                        features.append('NA')

                elif hasattr(m, 'feature_importance'):
                    if feature_names is not None:
                        # Check for decision trees feature importance
                        tp = np.argmax(m.feature_importance)
                    
                        # Match with corresponding feature
                        features.append(f'{feature_names[tp]}')
                    
                    else:
                        features.append('NA')
                else:
                    features.append('NA')
            else:
                features.append('NA')
        
        # Record result
        result = pd.DataFrame(
                list(zip(names, train_accuracies, accuracies, parameters, 
                         training_times, features)),
                columns=['Model', 'Train Accuracy', 'Test Accuracy', 
                         'Best Parameter', 'Train Time', 'Top Predictor'])
        
        # Show plot if desired
        if show_plot:
            axes = MLModels.plot_results(methods)
        else:
            axes = None
        
        # Display result
        if show_top:
            display(HTML('<center>' + result.to_html(index=False) 
                         + '</center>'))
        else:
            display(HTML('<center>' + result[[
                                    'Model', 'Test Accuracy', 
                                    'Best Parameter']].to_html(index=False) 
                         + '</center>'))
        
        return result, axes

    def plot_permutation_importance(
            est, X, y, feature_names=None, num_feat=10, scorer=None,
            n_repeats=10,
            random_state=1337):
        """
        Plot the feature importance via permutation importance method

        Generates the permutation importance plot given the trained model,
        input data and labels.

        Parameters
        ----------
        est : trained sklearn estimator
            Trained model to be inspected
        X : array-like
            Input data
        y : array-like
            Label data
        feature_names : list of str or array-like, default=None
            List of name of features
        num_feat : int, default=10
            Number of features to be included in the plot
        scorer : str, default=None
            Scorer to be used. See MLModels.scorers() for full list.
        n_repeats : int, default=10
            Number of times to permute a feature
        random_state : int, default=1337
            Random state of the permutation randomizer
        """
        # Get scorer
        if scorer is not None:
            scoring = make_scorer(MLModels.scorers()[scorer])
        else:
            scoring = None

        # Get importances
        importances = permutation_importance(
                est, X, y, n_repeats=n_repeats,
                random_state=1337, scoring=scoring)
        importances_mean = importances['importances_mean']

        # Set num feat depending on the number of features
        num_feat = min([len(importances_mean), num_feat])

        # Normalize importances
        importances_mean = importances_mean / importances_mean.sum()

        # Get indices
        indices = importances_mean.argsort()[::-1][:num_feat]

        # Initialize figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Generate bar graph of importances
        if feature_names is not None:
            ax.barh(np.array(feature_names)[indices][::-1],
                    importances_mean[indices][::-1])
        else:
            ax.barh(range(num_feat), importances_mean[indices[::-1]])

        # Set axis labels
        ax.set_xlabel('Importances', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return ax

    def plot_partial_dependence(est, X, features, feature_names):
        """
        Plot the partial dependence of features wrt the estimator
        
        Generates the partial dependence relation plots of the given list of
        features with respect to the given estimator.
        
        Parameters
        ----------
        est : trained sklearn-like estimator
            Trained model to be inspected
        X : array-like
            The dataset in which partial dependence is to be inspected,
            usually this is the train set
        features : list of str or tuple of str
            Name of features (or pair of features) to be inspected
        feature_names : list of str
            Name of features of the dataset
        
        Returns
        -------
        fig, axes : matplotlib figure and axes
            Figure and axes of the partial dependence plots
            
        Examples
        --------
        First run some models to get the optimal parameters
        >>> m = MLModels.run_classifier(X, label,
                n_trials=100, use_methods=['SVM (L1)', 'GB Classifier'])
        >>> res = MLModels.summarize(m, feature_names, show_plot=True, 
                                     show_top=True)
        
        Then train the estimator using the optimal parameter
        >>> est = m['GB Classifier'].model(max_depth=3).fit(X, label)
        
        Plot the partial dependence using the selected `features`
        >>> fig, axes = MLModels.plot_partial_dependence(est, X, features, 
                                                         feature_names)
        """
        # Set X_data as X in a pandas data frame form
        X_data = pd.DataFrame(X, columns=feature_names)
        
        # Get number of features
        num_features = len(features)

        # Compute for the number of columns and the number of rows
        ncols = (3 if num_features >= 3 else num_features)
        nrows = ceil(num_features/3)

        # Initialize figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), 
                                 sharey=True)

        # Iterate through all features and axes
        for ax, feature in zip(axes.flatten(), features):
            # Get partial dependence values
            ave_predictions, values = partial_dependence(
                    est, X_data, feature, grid_resolution=20)

            # Check if len of feature is 1
            if isinstance(feature, str):
                # Plot points
                ax.plot(values[0], ave_predictions[0], lw=3.0)

                # Get ylim
                ylim = np.array(ax.get_ylim())

                # Plot decile tickmarks
                ax.vlines(np.quantile(X_data[feature], 
                                      np.arange(0.1, 1.0, 0.10)), 
                          ylim[0], 
                          ylim[0] + 0.05*np.diff(ylim))

                # Set ylim
                ax.set_ylim(ylim)

                # Add axis labels
                if len(ax.get_yticklabels()) != 0:
                    ax.set_ylabel('Partial Dependence', fontsize=14)
                ax.set_xlabel(feature, fontsize=14)

            # Check if len of feature is 2
            elif len(feature) == 2:
                # Remove sharey
                [ax.get_shared_y_axes().remove(axis) for axis in axes.ravel()]

                # Create mesh grid
                Y, X = np.meshgrid(values[1], values[0])

                # Get xlim and ylim
                ylim = np.array([np.min(Y), np.max(Y)])
                xlim = np.array([np.min(X), np.max(X)])

                # Set axis limits
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                # Plot points
                ax.contourf(X, Y, ave_predictions[0])
                cs = ax.contour(X, Y, ave_predictions[0], colors='black')

                # Add contour label
                ax.clabel(cs, fontsize=10, colors='black', fmt='%1.3f')

                # Get xlim and ylim
                ylim = np.array([np.min(Y), np.max(Y)])
                xlim = np.array([np.min(X), np.max(X)])

                # Plot decile tickmarks
                ax.vlines(np.quantile(X_data[feature[0]], 
                                      np.arange(0.1, 1.0, 0.10)), 
                          ylim[0], ylim[0] + 0.05*np.diff(ylim))
                ax.hlines(np.quantile(X_data[feature[1]], 
                                      np.arange(0.1, 1.0, 0.10)), 
                          xlim[0], xlim[0] + 0.025*np.diff(xlim))

                # Set axis limits
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                # Set axis labels
                ax.set_xlabel(feature[0])
                ax.set_ylabel(feature[1])

        # Remove unused axes
        for i in range(num_features, ncols*nrows):
            # Off axis
            axes.flatten()[i].axis('off')
        
        return fig, axes

def plot_hyperparameter_accuracies(all_scores, all_std, params):
    """Plot score versus hyperparameter graph"""
    # Initialize figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Check the length of parameters
    if len(params.keys()) == 1:
        # Plot accuracies
        ax.plot(list(params.values())[0], all_scores, lw=3.0,
                color='tab:blue')

        # Plot stds
        ax.fill_between(list(params.values())[0],
                        np.array(all_scores) - np.array(all_std),
                        np.array(all_scores) + np.array(all_std),
                        alpha=0.2, color='tab:blue')

        # Set axis labels
        ax.set_xlabel(list(params.keys())[0], fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    elif len(params.keys()) == 2:
        # Get settings
        X = np.array(list(itertools.product(*params.values())))[:, 0]
        Y = np.array(list(itertools.product(*params.values())))[:, 1]

        # Plot accuracies as contour
        ax.tricontourf(X, Y, all_scores)
        cs = ax.tricontour(X, Y, all_scores, colors='black')

        # Set contour labels
        ax.clabel(cs, fontsize=10, colors='black', fmt='%.4f')

        # Set axis labels
        ax.set_xlabel(list(params.keys())[0], fontsize=14)
        ax.set_ylabel(list(params.keys())[1], fontsize=14)

    # Show figure
    plt.show()

def tune_model(X, label, task, model, params, n_trials, tree_rs=1337,
             write_to_file=None, to_plot=False, scorer=None, scaling=None):
    """
    Return tuned model and params for a given model, dataset, and parameters

    Trains and test iterations of a chosen model using the specified
    parameters. Prints the result everytime a higher accuracy is obtained.
    Return the optimized model and a summary of optimal hyper parameters as a
    dictionary. Can be specified to write to a file if desired. Can also be
    specified to plot score vs hyper parameters graph.

    Parameters
    ----------
    X : array-like
        Input features data
    label : array-like
        Labels of each data
    task : str
        Input 'Classification' for Classification task.
        'Regression' for regression task.
    model : str
        Model to be tuned. Execute MLModels.list_all_methods() for the
        possible choices.
    params : dict
        Parameters to be hypertuned. Keys are the name of the parameter,
        values are the list of values.
    n_trials : int
        Number of splitting trials to run per hyper parameter setting
    tree_rs : int
        Random state to be used for tree models.
    write_to_file : str
        Filepath / filename to which the result will be written.
    to_plot : bool
        Set to True if score vs parameter plots is desired. Works only when
        the number of hyper parameters is at most 2.
    scorer : str, default=None
        Set specified scorer if desired
    scaling : str, default=None
        Set specified scaling if desired

    Returns
    -------
    res : dict
        Dictionary containing the optimal model and hypertuned parameter
        result summary

    Examples
    --------
    Tune a GradientBoostingClassifier given the dataset X, and y and
    hyper tuning parameters. Store the result on a text file 'tune_gbc.txt':

    >>> res = tune_model(X, y, 'Classification', 'GB Classifier',
                         params={'max_depth': list(range(1, 11)),
                                 'n_estimators': [50, 100, 200],
                                 'learning_rate': [0.1, 0.5, 1]},
                         write_to_file='tune_gbc.txt')
    """
    # Get model
    clf = MLModels.all_methods(tree_rs=tree_rs)[task][model].model

    # Initialize score variables
    cur_score = 0
    all_scores = []
    all_std = []

    # Initialize progress bar
    pb = tqdm(total=(np.product(list(map(lambda x: len(x), params.values())))
                     *n_trials))

    # Iterate through hyperparameters
    for param in itertools.product(*params.values()):
        # Get current parameters
        cur_param = dict(zip(params.keys(), param))

        # Initialize score container
        scores = []

        # Perform splitting trials
        for i in range(n_trials):
                # Split the dataset
                X_train, X_test, y_train, y_test = train_test_split(
                        X, label, random_state=i)

                # Perform scaling
                if scaling is not None:
                    # Check if power scaling
                    if scaling == 'power':
                        # Check if strictly positive features
                        if (X_train <= 0).sum() == 0:
                            # Set scaler
                            scaler = MLModels.scalers()['power']

                            # Set method to box cox
                            scaler.set_params(method='box-cox')
                        else:
                            scaler = MLModels.scalers()[scaling]
                    else:
                        # Set scaler based on user option
                        scaler = MLModels.scalers()[scaling]

                    # Fit to train dataset
                    scaler.fit(X_train)

                    # Transform train and test dataset
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)

                # Initialize model
                cur_clf = clf(**cur_param)

                # Train model
                cur_clf.fit(X_train, y_train.ravel())

                # Get score
                if scorer is not None:
                    scores.append(MLModels.scorers()[scorer](
                            y_test, cur_clf.predict(X_test)))
                else:
                    scores.append(cur_clf.score(X_test, y_test))

                # Update progress bar
                pb.update(1)

        # Get mean score
        score = np.mean(scores)
        all_scores.append(score)

        # Get standard deviation
        std = np.std(scores)
        all_std.append(std)

        # Print if current max
        if score > cur_score:
            # Print current max paramters
            to_print = ('params:{} \tscore: {:.6f}\tstd: {:.6f}'.format(
                    cur_param, score, std))
            print(to_print)

            # Update current max score
            cur_score = score

            # Update current optimal parameter result
            res = {'Model': model,
                   'Accuracy': '{:.2f}%'.format(score*100),
                   'Best Parameter': cur_param,
                   'Classifier': cur_clf,
                   'acc': np.round(100*score, 2),
                   'std': std
                  }

            # Write to text file the current result if desired
            if write_to_file:
                with open(write_to_file, 'a') as f:
                    f.write(to_print+'\n')

    # Delete progress bar
    del pb

    # Check if plotting is desired
    if to_plot:
        plot_hyperparameter_accuracies(all_scores, all_std, params)

    return res

def tune_gbc(X, label, max_depths, learning_rates, n_estimators=100, 
             random_state=1337, write_to_txt=None):
    """
    Return tuned parameters for GB Classifier given dataset and parameters
    
    Trains and test iterations of GradientBoosting Classifier using the 
    specified max depths and learning rates, with a set `n_estimators`.
    Prints the result everytime a higher accuracy is obtained. Returns a
    summary of optimal hyperparameters as a dictionary.
    
    Paramters
    ---------
    X : array-like
        Input features data
    label : array-like
        Labels of each data
    max_depths : list-like
        List of max depths to be used in hyper parameter tuning
    learning_rates : list-like
        List of learning rates to be used in hyper parameter tuning
    n_estimator : int
        `n_estimator` parameter to be set in the GradientBoostingClassifier
    write_to_txt : str
        Specify text file name if results are to be written in a file
    
    Returns
    -------
    res : dict
        Summary of the resulting hyperparameter tuning
        
    Examples
    --------
    Hyper parameter tune data `X`, with labels `label` using max depth from 1
    to 10, and learning rates 0.10 to 2.50 with 0.10 increments.
    
    >>> max_depths = list(range(1, 10))
    >>> learning_rates = list(np.arange(0.10, 2.5, 0.10))
    >>> gbc_res = tune_gbc(X, label, max_depths, learning_rates)
    """
    # Initialize variables
    cur_score = 0
    
    # Initialize progress bar
    pb = tqdm(total=len(learning_rates)*len(max_depths))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
            X, label, random_state=1337)
    
    # Iterate through hyperparameters
    for learning_rate in learning_rates:
        # Set progress bar description
        pb.set_description(f'Learning Rate: {np.round(learning_rate, 4)}')
        
        # Iterate through max depths
        for max_depth in max_depths:
            # Train model
            gbm = GradientBoostingClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate, 
                    max_depth=max_depth, random_state=random_state).fit(
                    X_train, y_train.ravel())

            # Get score
            score = gbm.score(X_test, y_test)

            # Print if current max
            if score > cur_score:
                # Print current max paramters
                to_print = (f'learning rate: {np.round(learning_rate, 5)}\tma'
                      f'x_depth: {max_depth}\tscore: {score}')
                print(to_print)

                # Update current max score
                cur_score = score

                # Create parameter dictionary
                params = dict(learning_rate=learning_rate,
                              max_depth=max_depth)

                # Update current optimal parameter result
                res = {'Model': 'Gradient Boost',
                       'Accuracy': '{:.2f}%'.format(score*100),
                       'Best Parameter': params,
                       'Top Predictor': 'NA',
                       'acc': np.round(100*score, 2),
                      }

                # Write to text file the current result if desired
                if write_to_txt:
                    with open(write_to_txt, 'a') as f:
                        f.write(to_print+'\n')

            # Update progress bar
            pb.update(1)

    # Delete progress bar
    del pb
    
    return res

def tune_RF(X, label, max_depths, n_estimators, max_features='sqrt', 
             random_state=1337, write_to_txt=None):
    """
    Return tuned parameters for GB Classifier given dataset and parameters

    Trains and test iterations of GradientBoosting Classifier using the 
    specified max depths and learning rates, with a set `n_estimators`.
    Prints the result everytime an higher accuracy is obtained. Returns a
    summary of optimal hyperparameters as a dictionary.

    Paramters
    ---------
    X : array-like
        Input features data
    label : array-like
        Labels of each data
    max_depths : list-like
        List of max depths to be used in hyper parameter tuning
    n_estimators : list-like
        List of n_estimators to be used in hyper parameter tuning
    max_features : int or str
        `max_features` parameter to be set in the GradientBoostingClassifier
    write_to_txt : str
        Specify text file name if results are to be written in a file
    
    Returns
    -------
    res : dict
        Summary of the resulting hyperparameter tuning
        
    Examples
    --------
    Hyper parameter tune data `X`, with labels `label` using max depth from 1
    to 10, and learning rates 0.10 to 2.50 with 0.10 increments.
    
    >>> max_depths = list(range(1, 10))
    >>> n_estimators = list(np.arange(0.10, 2.5, 0.10))
    >>> gbc_res = tune_gbc(X, label, max_depths, n_estimators)
    """
    # Initialize variables
    cur_score = 0
    
    # Initialize progress bar
    pb = tqdm(total=len(n_estimators)*len(max_depths))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
            X, label, random_state=1337)

    # Set max depth
    if isinstance(max_features, str):
        if max_features == 'sqrt':
            max_features = int(np.sqrt(X_train.shape[1]))
            print(max_features)
    
    # Iterate through hyperparameters
    for n_estimator in n_estimators:
        # Set progress bar description
        pb.set_description(f'N estimator: {np.round(n_estimator, 4)}')
        
        # Iterate through max depths
        for max_depth in max_depths:
            # Train model
            rf = RandomForestClassifier(
                    n_estimators=n_estimator, max_features=max_features, 
                    max_depth=max_depth, random_state=random_state).fit(
                    X_train, y_train.ravel())

            # Get score
            score = rf.score(X_test, y_test)
            
            # Get max feature importance
            max_index = rf.feature_importances_.argsort()[-1]

            # Print if current max
            if score > cur_score:
                # Print current max paramters
                to_print = (f'n estimator: {np.round(n_estimator, 5)}\tma'
                      f'x_depth: {max_depth}\tscore: {score}')
                print(to_print)
                
                # Update current max score
                cur_score = score
                
                # Create parameter dictionary
                params = dict(n_estimator=n_estimator,
                              max_depth=max_depth)
                
                # Update current optimal parameter result
                res = {'Model': 'Random Forest', 
                       'Accuracy': '{:.2f}%'.format(score*100),
                       'Best Parameter': params,
                       'Top Predictor': max_index,
                       'acc': np.round(100*score, 2),
                      }
                
                # Write to text file the current result if desired
                if write_to_txt:
                    with open(write_to_txt, 'a') as f:
                        f.write(to_print+'\n')
                
            # Update progress bar
            pb.update(1)

    # Delete progress bar
    del pb
    
    return res


class KNN(MLModels):
    _setting_name = 'n_neighbors'
    _setting_label = 'n_\text{neighbors}'
    log_plot = False

    def __init__(self, neighbor_setting):
        super().__init__()
        self._setting = neighbor_setting


class KNNClassifier(KNN):
    model = KNeighborsClassifier


class KNNRegressor(KNN):
    model = partial(KNeighborsRegressor, algorithm='kd_tree')

class LinearRegressor(MLModels):
    model = None
    _setting_name = 'alpha'
    _setting_label = r'\alpha'
    log_plot = True

    def __init__(self, alpha):
        super().__init__()
        self._setting = alpha

class MLPC(LinearRegressor):
    model = partial(MLPClassifier)

class MLPR(LinearRegressor):
    model = partial(MLPRegressor)

class LassoRegressor(LinearRegressor):
    model = partial(Lasso, max_iter=10000)

class RidgeRegressor(LinearRegressor):
    model = partial(Ridge, max_iter=10000)

# Elastic Net
class EN(LinearRegressor):
    model = partial(ElasticNet, max_iter=10000)

# Gaussian Process Regression
class GPRegressor(LinearRegressor):
    model = partial(GaussianProcessRegressor)
    
class SVMRegressor(MLModels):
    _setting_name = 'C'
    _setting_label = 'C'
    log_plot = True
    def __init__(self, C, kernel='linear'):
        self._setting = C
        self.model = partial(SVR, kernel=kernel)

class LinearClassifier(LinearRegressor):
    model = None
    _setting_name = 'C'
    _setting_label = 'C'
    log_plot = True

    def __init__(self, C, reg='l2'):
        self._setting = C
        self._init_model(reg)

    def _init_model(self, reg):
        raise NotImplementedError()

class NonLinearClassifier(MLModels):
    model = None
    _setting_name = 'C'
    _setting_label = 'C'
    log_plot = True
    
    def __init__(self, C, kernel, degree=3.0):
        self._setting = C
        self._init_model(kernel, degree=3.0)

class LogisticRegressor(LinearClassifier):
    def _init_model(self, reg):
        self.model = partial(LogisticRegression,
                             solver='liblinear', penalty=reg)

class LinearSVM(LinearClassifier):
    def _init_model(self, reg):
        self.model = partial(LinearSVC, loss='squared_hinge',
                             dual=False, penalty=reg)

class NonLinearSVM(NonLinearClassifier):
    def _init_model(self, kernel, degree):
        self.model = partial(SVC, kernel=kernel, degree=degree)

# SGD Models
class SGDModels(MLModels):
    model = None
    _setting_name = 'alpha'
    _setting_label = r'\alpha'
    log_plot = True

    def __init__(self, alpha, reg, loss):
        self._setting = alpha
        self._init_model(reg, loss)

    def _init_model(self, reg, loss):
        raise NotImplementedError()

class SGDC(SGDModels):
    def _init_model(self, reg, loss):
        self.model = partial(SGDClassifier, penalty=reg, loss=loss)

class SGDR(SGDModels):
    def _init_model(self, reg, loss):
        self.model = partial(SGDRegressor, penalty=reg, loss=loss)

# Decision Tree Classifiers
class TreeClassifiers(MLModels):
    model = None
    _setting_name = 'max_depth'
    _setting_label = r'Max Depth'
    log_plot = False

    def __init__(self, max_depth, tree_rs):
        super().__init__()
        self._setting = max_depth
        self._init_model(tree_rs)

# Basic Decision Tree
class DecisionTree(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(DecisionTreeClassifier, random_state=tree_rs)
        
# Random Forest
class RFClassifier(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(RandomForestClassifier, random_state=tree_rs,
                             n_jobs=-1)

# Extra Trees
class ETClassifier(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(ExtraTreesClassifier, random_state=tree_rs, 
                             n_jobs=-1)

# GB Classifier
class GBClassifier(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(GradientBoostingClassifier, random_state=tree_rs)
        
# Hist GBC
class HistGBC(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(HistGradientBoostingClassifier, 
                             random_state=tree_rs)

# AdaBoostDT Classifier
class ABDTClassifier(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(
            lambda max_depth, random_state: AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=max_depth), 
                random_state=random_state), random_state=tree_rs)

# Decision Tree Regressor
class TreeRegressors(MLModels):
    model = None
    _setting_name = 'max_depth'
    _setting_label = r'Max Depth'
    log_plot = False
    
    def __init__(self, max_depth, tree_rs):
        super().__init__()
        self._setting = max_depth
        self._init_model(tree_rs)
        
# Basic Decision Tree Regressor
class DTRegressor(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(DecisionTreeRegressor, random_state=tree_rs)
        
# Random Forest Regressor
class RFRegressor(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(RandomForestRegressor, random_state=tree_rs,
                             n_jobs=-1)
class ETRegressor(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(ExtraTreesRegressor, random_state=tree_rs, 
                             n_jobs=-1)

# GB Regressor
class GBRegressor(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(GradientBoostingRegressor, random_state=tree_rs)

# Histogram GBR
class HistGBR(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(HistGradientBoostingRegressor, 
                             random_state=tree_rs)

# AdaBoostDT Regressor
class ABDTRegressor(TreeClassifiers):
    def _init_model(self, tree_rs):
        self.model = partial(
            lambda max_depth, random_state: AdaBoostRegressor(
                DecisionTreeRegressor(max_depth=max_depth), 
                random_state=random_state), random_state=tree_rs)
        
# Naive Bayes
class NaiveBayesClassifier(MLModels):
    model = None
    _setting_name = 'alpha'
    _setting_label = r'\alpha'
    log_plot = True

    def __init__(self, alpha):
        super().__init__()
        self._setting = alpha

# Advance GBM Models
if adv_gbm:
    # XGBoost
    class XGBC(TreeClassifiers):
        def _init_model(self, tree_rs):
            self.model = partial(XGBClassifier, random_state=tree_rs)
    class XGBR(TreeClassifiers):
        def _init_model(self, tree_rs):
            self.model = partial(XGBRegressor, random_state=tree_rs)

    # LightGBM
    class LGBMC(TreeClassifiers):
        def _init_model(self, tree_rs):
            self.model = partial(LGBMClassifier, random_state=tree_rs)
    class LGBMR(TreeClassifiers):
        def _init_model(self,tree_rs):
            self.model = partial(LGBMRegressor, random_state=tree_rs)

    # CatBoost
    class CBClassifier(TreeClassifiers):
        def _init_model(self, tree_rs):
            self.model = partial(CatBoostClassifier, random_state=tree_rs, 
                                 verbose=0)
    class CBRegressor(TreeClassifiers):
        def _init_model(self, tree_rs):
            self.model = partial(CatBoostRegressor, random_state=tree_rs,
                                 verbose=0)

# MultinomialNB, ComplementNB, BernoulliNB
class MNB(NaiveBayesClassifier):
    model = MultinomialNB

class CNB(NaiveBayesClassifier):
    model = ComplementNB

class BNB(NaiveBayesClassifier):
    model = BernoulliNB

# Voting Models
class VotingModel(MLModels):
    # Intitialize settings
    _setting = [None]
    _setting_name = None

    def __init__(self, models, params, task, voting='hard', weights=None):
        """
        Initialize Voting model using selected models and parameters

        Parameters
        ----------
        self : MLModels object
            Voting model object
        models : list of str
            List of str to be used in creating the ensemble.
            See MLModels.list_all_methods() for possible options.
        params : list of dict
            List of dictionary of hyperparameters of the selected model.
        task : str
            Set as 'Classification' or 'Regression' depending on the task
        voting : 'hard' or 'soft'
            Set voting condition
        weights : array like, default=None
            Set pre-defined weights of voting power of each classifier
        """
        # Initialize classifier container
        ests = []

        # Iterate through all selected models and corresponding parameters
        for m, args in zip(models, params):
            ests.append((m, MLModels.all_methods()[task][m].model()
                                    .set_params(**args)))

        # Set as model estimator
        self.ests = ests

        # Initialize model
        if task == 'Classification':
            self.model = VotingClassifier(ests, voting=voting,
                                          weights=weights)
            self.task = task
        elif task == 'Regression':
            self.model = VotingRegressor(ests, voting=voting, weights=weights)
            self.task = task

    def fit(self, X, y):
        """Fit the estimators using the input `X` and target `y` data"""
        self.model.fit(X, y)

    def predict(self, X):
        """Return model predictions given `X` data"""
        return self.model.predict(X)

    def score(self, X, y):
        """Return default accuracy setting from the given data"""
        return self.model.score(X, y)

    def summary(self):
        """Print a summary of train and test result of the voting model"""
        print("Voting Model Performance Summary")
        print("--------------------------------")
        print(f"Task: {self.task}\n")
        print("Training Accuracy: {:.2f}%".format(
                self.training_accuracy[0]*100))
        print("Test Accuracy: {:.2f}%".format(self.test_accuracy[0]*100))
        print("Training Time: {:.4f} secs".format(self.train_time[0]))