import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
    precision_score,
    roc_auc_score,
)
from IPython.display import display
from scipy.stats import ttest_ind, chi2_contingency, t, pointbiserialr, mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_iqr_and_outliers(dataframe, column):
    """
    Calculates the interquartile range (IQR) and finds outliers in a column of a DataFrame. Prints number of outliers,
    percentage of outliers in the dataset, minimum and maximum value among the outliers.
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    column (str): The column name for which to calculate the IQR and outliers.
    """
    # Calculate Q1, Q3, and IQR
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = dataframe[
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]

    # Calculate outlier statistics
    outlier_count = len(outliers)
    total_count = len(dataframe)
    outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0
    outlier_min = outliers[column].min() if not outliers.empty else None
    outlier_max = outliers[column].max() if not outliers.empty else None

    print(f"Number of outliers: {outlier_count}")
    print(f"Percentage of outliers: {outlier_percentage:.2f}%")
    print(f"Minimum outlier value: {outlier_min}")
    print(f"Maximum outlier value: {outlier_max}")


def missing_values_summary(dataframe):
    """
    Calculates the count and percentage of missing values in a pandas DataFrame.
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    Returns:
    pd.DataFrame with counts and percentages of missing values for each column.
    """
    missing_count = dataframe.isnull().sum()
    missing_percentage = (missing_count / len(dataframe)) * 100

    summary_df = pd.DataFrame(
        {"Missing Count": missing_count, "Missing Percentage": missing_percentage}
    ).sort_values(by="Missing Count", ascending=False)

    return summary_df


def compute_percentage(
    dataframe, groupby_cols, count_col_name="count", percentage_col_name="percentage"
):
    """
    Compute percentages for counts grouped by specified columns.
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    groupby_cols (list): List of column names to group by.
    count_col_name (str): Name of the count column in the resulting DataFrame.
    percentage_col_name (str): Name of the percentage column in the resulting DataFrame.
    Returns:
    pd.DataFrame: A new DataFrame with counts and percentages for each group.
    """
    percentage_df = (
        dataframe.groupby(groupby_cols).size().reset_index(name=count_col_name)
    )
    total_per_group = percentage_df.groupby(groupby_cols[:-1])[
        count_col_name
    ].transform("sum")
    percentage_df[percentage_col_name] = (
        percentage_df[count_col_name] / total_per_group
    ) * 100

    return percentage_df


def tune_hyperparameters(
    model,
    param_distributions,
    X_train,
    y_train,
    scoring="recall",
    n_iter=100,
    cv=5,
    random_state=50,
):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    Parameters:
        model (estimator): The machine learning model to tune.
        param_distributions (dict): Dictionary of hyperparameter distributions to sample from.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training labels.
        scoring (str): Scoring metric to optimize. Default is 'recall'.
        n_iter (int): Number of parameter settings to sample. Default is 100.
        cv (int): Number of cross-validation folds. Default is 5.
        random_state (int): Random seed for reproducibility.
    Returns:
        search (RandomizedSearchCV): The fitted RandomizedSearchCV object.
    """
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def show_top_results(search, dimensions, top_n=10):
    """
    Display the top N hyperparameter tuning results sorted by score.
    Parameters:
        search (RandomizedSearchCV): Fitted RandomizedSearchCV object.
        dimensions (list): List of hyperparameter names to display.
        top_n (int): Number of top rows to display. Default is 10.
    Returns:
        results_df (pd.DataFrame): Full DataFrame of cross-validation results.
    """
    results_df = pd.DataFrame(search.cv_results_)
    top_results = (
        results_df[dimensions + ["mean_test_score"]]
        .sort_values(by="mean_test_score", ascending=False)
        .head(top_n)
    )
    display(top_results)
    return results_df


def plot_parallel_coordinates(results_df, dimensions):
    """
    Plot parallel coordinates to visualize hyperparameter tuning results.
    Parameters:
        results_df (pd.DataFrame): DataFrame containing tuning results (from show_top_results).
        dimensions (list): List of hyperparameter columns to include in the plot.
    Returns:
        None: Displays a plotly interactive plot.
    """
    for col in dimensions:
        if results_df[col].dtype == "object":
            results_df[col] = results_df[col].astype("category").cat.codes

    fig = px.parallel_coordinates(
        results_df,
        dimensions=dimensions + ["mean_test_score"],
        color="mean_test_score",
        color_continuous_scale=px.colors.sequential.Bluered,
    )
    fig.show()


def evaluate_cross_validation(model, X_train, y_train, cv=5, scoring="recall"):
    """
    Evaluate a model using cross-validation and print the score summary.
    Parameters:
        model (estimator): The model to evaluate.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training labels.
        cv (int): Number of cross-validation folds. Default is 5.
        scoring (str): Scoring metric. Default is 'recall'.
    Returns:
        cv_scores (np.ndarray): Array of cross-validation scores.
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"Cross-validation (CV) {scoring.title()} Scores: {cv_scores}")
    print(f"Mean CV {scoring.title()}: {cv_scores.mean():.4f}")
    print(f"Std Dev of CV {scoring.title()}: {cv_scores.std():.4f}")
    return cv_scores


def evaluate_on_validation(model, X_val, y_val):
    """
    Evaluate the model on a validation set and print common metrics.
    Parameters:
        model (estimator): Trained model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): True labels for validation set.
    Returns:
        y_pred (np.ndarray): Predicted labels.
    """

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print("\nValidation (or Test) Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return y_pred


def plot_confusion(y_true, y_pred, model_name="Model"):
    """
    Plot a confusion matrix heatmap.
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Title of the plot. Default is "Model".
    Returns:
        None: Displays the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="2.0f",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def evaluate_models_cv(
    models, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=50), figsize=(18, 10), title_prefix="Cross-Validation"
):
    """
    Evaluate multiple models using cross-validation and plot boxplots of evaluation metrics and fit times.
    Parameters:
    - models (dict): Dictionary with model names as keys and estimators as values.
    - X (np.ndarray or pd.DataFrame): Feature matrix.
    - y (np.ndarray or pd.Series): Target vector.
    - cv (int): Number of cross-validation folds.
    - figsize (tuple): Figure size for the plots.
    - title_prefix (str): Prefix for the subplot titles.
    Returns:
    pd.DataFrame: Cross-validation results for each model, metric, and timing.
    """

    scoring = {
        "Accuracy": "accuracy",
        "Recall": make_scorer(recall_score),
        "Precision": make_scorer(precision_score),
        "F1 Score": make_scorer(f1_score),
        "ROC AUC": "roc_auc"
    }

    all_results = []

    for model_name, model in models.items():
        cv_result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            return_estimator=False,
        )

        for i in range(cv):
            all_results.append({
                "Model": model_name,
                "Accuracy": cv_result["test_Accuracy"][i],
                "Recall": cv_result["test_Recall"][i],
                "Precision": cv_result["test_Precision"][i],
                "F1 Score": cv_result["test_F1 Score"][i],
                "ROC AUC": cv_result["test_ROC AUC"][i],
                "Fit Time (s)": cv_result["fit_time"][i],
                "Score Time (s)": cv_result["score_time"][i],
            })

    df = pd.DataFrame(all_results)

    metrics_to_plot = ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC", "Fit Time (s)"]

    plt.figure(figsize=figsize)
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x="Model", y=metric, data=df, hue = "Model")
        plt.title(f"{title_prefix} {metric}")

    plt.tight_layout()
    plt.show()

    return df

def get_average_prediction_time(model, X, repeats=10):
    times = []
    for _ in range(repeats):
        start = time.time()
        model.predict(X)
        end = time.time()
        times.append(end - start)
    return sum(times) / repeats


def evaluate_models_validation(
    models, X_val, y_val, figsize=(10, 6), title="Validation Set Performance"
):
    """
    Evaluate multiple models on a validation dataset and plot their performance metrics including prediction time.
    Parameters:
    - models (dict): Dictionary of model name → model instance (already fitted).
    - X_val (np.ndarray or pd.DataFrame): Validation feature set.
    - y_val (np.ndarray or pd.Series): True labels for validation set.
    - figsize (tuple): Size of the plot.
    - title (str): Title of the plot.
    Returns:
    pd.DataFrame: DataFrame with performance scores per model including prediction time.
    """
    metrics = {
        "Accuracy": accuracy_score,
        "Recall": recall_score,
        "Precision": precision_score,
        "F1 Score": f1_score,
        "ROC AUC": roc_auc_score,
    }

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_val)
        inference_time = get_average_prediction_time(model, X_val)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_val)[:, 1]
            except:
                pass

        scores = {}
        for metric_name, func in metrics.items():
            if metric_name == "ROC AUC" and y_proba is not None:
                score = func(y_val, y_proba)
            elif metric_name == "ROC AUC":
                score = np.nan
            else:
                score = func(y_val, y_pred)
            scores[metric_name] = score

        scores["Time (s)"] = inference_time
        scores["Model"] = name
        results.append(scores)

    df = pd.DataFrame(results)

    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x="Metric",
        y="Score",
        hue="Model",
        data=df_melted,
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title(title)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.grid()

    for p in ax.patches:
        height = p.get_height()
        if height > 0 and height <= 1.1:  # Annotate only the metrics, not time
            ax.annotate(
                f'{height:.2f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=7,
                xytext=(0, 3),
                textcoords='offset points', 
                rotation = 45
            )

    plt.tight_layout()
    plt.show()

    return df


def compare_means_ttest(X, y, feature):
    """
    Performs independent t-test on a continuous feature between two target classes.
    Returns t-statistic, p-value, and 95% confidence interval for the mean difference.
    """
    group1 = X[y == 1][feature]
    group0 = X[y == 0][feature]

    t_stat, p_value = ttest_ind(group1, group0)

    mean_diff = np.mean(group1) - np.mean(group0)
    se_diff = np.sqrt(
        np.var(group1, ddof=1) / len(group1) + np.var(group0, ddof=1) / len(group0)
    )

    df = len(group1) + len(group0) - 2
    ci_lower = mean_diff - t.ppf(0.975, df) * se_diff
    ci_upper = mean_diff + t.ppf(0.975, df) * se_diff

    print(f"{feature} T-statistic: {t_stat:.2f}, P-value: {p_value:.3f}")
    print(f"{feature} 95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")

    return t_stat, p_value, (ci_lower, ci_upper)

def compare_means_mannwhitney(X, y, feature, n_bootstraps=1000, ci=0.95, random_state=50):
    """
    Performs Mann–Whitney U test on a continuous feature between two target classes.
    Also estimates the confidence interval for the median difference using bootstrapping.
    Parameters:
    X: DataFrame with features.
    y: Target variable (binary: 0 and 1).
    feature: Feature to test.
    n_bootstraps: Number of bootstrap samples.
    ci: Confidence level for the interval.
    random_state: Random seed for reproducibility.
    Returns:
    u_stat: Mann–Whitney U statistic.
    p_value: P-value from the test.
    ci_lower, ci_upper: Confidence interval for the median difference.
    """
    group1 = X[y == 1][feature].values
    group0 = X[y == 0][feature].values

    u_stat, p_value = mannwhitneyu(group1, group0, alternative="two-sided")

    rng = np.random.default_rng(random_state)
    med_diffs = []

    for _ in range(n_bootstraps):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample0 = rng.choice(group0, size=len(group0), replace=True)
        med_diffs.append(np.median(sample1) - np.median(sample0))

    lower = np.percentile(med_diffs, (1 - ci) / 2 * 100)
    upper = np.percentile(med_diffs, (1 + ci) / 2 * 100)

    print(f"{feature} Mann–Whitney U-statistic: {u_stat:.2f}, P-value: {p_value:.3f}")
    print(f"{feature} Estimated {int(ci*100)}% CI for median difference: ({lower:.2f}, {upper:.2f})")

    return u_stat, p_value, (lower, upper)


def compare_proportions_chi2(X, y, binary_feature):
    """
    Performs chi-square test on a binary feature and returns p-value and 95% CI for difference in proportions.
    """
    contingency_table = pd.crosstab(X[binary_feature], y)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    prop1 = X[y == 1][binary_feature].mean()
    prop0 = X[y == 0][binary_feature].mean()

    n1 = len(X[y == 1])
    n0 = len(X[y == 0])

    se_diff = np.sqrt((prop1 * (1 - prop1) / n1) + (prop0 * (1 - prop0) / n0))
    diff = prop1 - prop0
    z_critical = 1.96
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff

    print(f"{binary_feature} Chi2 Statistic: {chi2_stat:.2f}, P-value: {p_value:.3f}")
    print(
        f"{binary_feature} 95% CI for Proportion Difference: ({ci_lower:.3f}, {ci_upper:.3f})"
    )

    return chi2_stat, p_value, (ci_lower, ci_upper)


def calculate_combined_correlations(
    X_train, y_train, numeric_features, categorical_features
):
    """
    Calculates Pearson, Spearman, and Point-Biserial correlations between features and target.
    Parameters:
        X_train (pd.DataFrame): Feature matrix.
        y_train (pd.Series): Target variable.
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.
    Returns:
        combined_corr_df (pd.DataFrame): Combined correlation table with types.
        corr_matrix (pd.DataFrame): Correlation matrix for visualization.
    """
    target_name = y_train.name
    X_train_with_target = X_train.copy()
    X_train_with_target[target_name] = y_train

    numeric_corr = X_train_with_target[numeric_features + [target_name]].corr(method="pearson")
    categorical_corr = X_train_with_target[categorical_features + [target_name]].corr(method="spearman")

    numeric_categorical_corr = []

    for num_col in numeric_features:
        for cat_col in categorical_features:
            corr, _ = pointbiserialr(X_train_with_target[num_col], X_train_with_target[cat_col])
            numeric_categorical_corr.append([num_col, cat_col, corr])

        corr, _ = pointbiserialr(X_train_with_target[num_col], y_train)
        numeric_categorical_corr.append([num_col, target_name, corr])

    for cat_col in categorical_features:
        corr, _ = pointbiserialr(X_train_with_target[cat_col], y_train)
        numeric_categorical_corr.append([cat_col, target_name, corr])

    numeric_categorical_corr_df = pd.DataFrame(
        numeric_categorical_corr, columns=["Variable 1", "Variable 2", "Correlation"]
    )
    numeric_categorical_corr_df["Type"] = "Point-Biserial"

    numeric_corr_df = numeric_corr.stack().reset_index(name="Correlation")
    numeric_corr_df["Type"] = "Pearson"
    numeric_corr_df.columns = ["Variable 1", "Variable 2", "Correlation", "Type"]

    categorical_corr_df = categorical_corr.stack().reset_index(name="Correlation")
    categorical_corr_df["Type"] = "Spearman"
    categorical_corr_df.columns = ["Variable 1", "Variable 2", "Correlation", "Type"]

    combined_corr_df = pd.concat(
        [numeric_corr_df, categorical_corr_df, numeric_categorical_corr_df],
        ignore_index=True,
    )
    combined_corr_df.sort_values(by=["Variable 1", "Variable 2"], inplace=True)

    corr_matrix = combined_corr_df.pivot_table(
        index="Variable 1", columns="Variable 2", values="Correlation"
    )
    corr_matrix = corr_matrix.where(pd.notna(corr_matrix), corr_matrix.T)

    return combined_corr_df, corr_matrix


class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols, target_col):
        self.group_cols = group_cols
        self.target_col = target_col

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group_cols)[self.target_col].median()
        self.overall_median_ = X[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        def impute(row):
            if pd.isna(row[self.target_col]):
                return self.group_medians_.get(
                    tuple(row[col] for col in self.group_cols),
                    self.overall_median_
                )
            return row[self.target_col]
        X[self.target_col] = X.apply(impute, axis=1)
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)