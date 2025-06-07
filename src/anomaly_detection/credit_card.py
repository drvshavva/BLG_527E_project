import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns

from src.models.one_class_gmm import OneClassGMM


class CreditCardFraudDetection:
    """Main class for credit card fraud detection with SHAP feature selection"""

    def __init__(self):
        self.data = None
        self.feature_names = None
        self.feature_importance_ranking = None
        self.results = {}

    def load_and_prepare_data(self, file_path):
        """
        Load and prepare the credit card fraud dataset
        The Credit Card dataset is used. 29 features are used except for the "time" feature.
        The features are scaled to the range [0,1].
        """
        try:
            print("Loading and preparing dataset...")

            # Load data
            self.data = pd.read_csv(file_path)

            # Remove 'Time' column and separate features from target
            if 'Time' in self.data.columns:
                self.data = self.data.drop('Time', axis=1)
                print("Excluded 'Time' feature as specified in methodology")

            # Separate features and target
            X = self.data.drop('Class', axis=1)
            y = self.data['Class']

            # Scale features to [0,1]
            print("Scaling features to [0,1] range...")
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            self.feature_names = X.columns.tolist()
            self.X = X_scaled
            self.y = y

            print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
            print(f"Class distribution: Normal={sum(y == 0)}, Fraud={sum(y == 1)}")

            return self.X, self.y

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Please ensure the file path is correct and the file exists.")
            print("You can download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data")
            return None, None

    def phase1_feature_selection(self, model_type='svm'):
        """
        The purpose of this phase is to rank the features of the dataset according to their importance using SHAP values.

        Phase 1: Feature selection using SHAP values
            1. Preparing the Dataset
            2. Training the Model
            3. Using SHAP KernelExplainer
            4. Calculating SHAP Values
            5. Determining Feature Importance
            6. Ranking the Features
        """
        print(f"\n=== Phase 1: Feature Selection with {model_type.upper()} ===")

        if self.data is None:
            raise ValueError("Must run load_and_prepare_data first")

        # Step 2: Train the model on entire dataset
        # For feature selection, an uncalibrated One-Class SVM or One-Class GMM model is trained on the entire dataset.
        # The model's hyperparameters are fixed (n_components=2 for GMM, nu=0.0025 for SVM, kernel="rbf", gamma=0.0310).
        print("Training model for feature selection...")
        if model_type == 'svm':
            # parameters from paper
            model = OneClassSVM(nu=0.0025, kernel='rbf', gamma=0.0310)
        elif model_type == 'gmm':
            model = OneClassGMM(n_components=2)
        else:
            raise ValueError("model_type must be 'svm' or 'gmm'")

        # Train on normal data only (Class == 0)
        normal_data = self.X[self.y == 0]
        model.fit(normal_data)

        # Step 3: Create SHAP explainer
        # The trained model and a sample of the dataset are given to the SHAP KernelExplainer object.
        # A default of 100 samples are taken using the shap.sample function.
        print("Creating SHAP explainer...")
        # number of samples 100 (from paper)
        background_data = shap.sample(self.X, 100)
        explainer = shap.KernelExplainer(model.decision_function, background_data)

        # Step 4: Calculate SHAP values
        # The shap_values function of the KernelExplainer object is called.
        # This function returns a two-dimensional array of SHAP (feature importance) values for each sample.
        print("Calculating SHAP values...")
        shap_sample = shap.sample(self.X, 100)  # todo: bunu kaldırıp tüm sampleları versek mi? uzun sürer ama
        shap_values = explainer.shap_values(shap_sample)

        # Step 5: Determine feature importance
        # The feature importance value of each feature is found by averaging the absolute values of each row of the SHAP array (i.e. each sample).
        print("Determining feature importance...")
        feature_importance = np.abs(shap_values).mean(0)

        # Step 6: Rank features
        # he features are ranked according to their calculated average absolute SHAP values.
        # This ranking is done from the feature with the highest SHAP value to the lowest.
        # The results of this step are lists that give the order of importance of the features.
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        })
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        self.feature_importance_ranking = feature_importance_df['feature'].tolist()

        print("Feature importance ranking (top 15):")
        for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
            print(f"{i + 1:2d}. {row['feature']:>4s}: {row['importance']:.4f}")

        return self.feature_importance_ranking, feature_importance_df

    def phase2_model_evaluation(self, model_type='svm', feature_counts=[3, 5, 7, 10, 15, 29]):
        """
        In this phase, models are created using different numbers of features according to the feature importance
        ranking determined in Phase 1 and their performances are evaluated.
        Phase 2: Model evaluation with different feature counts
            1. Creating a Feature Set
            2. Cross-Validation
            3. Single-Class Training Data Preparation
            4. Training and Testing the Model
            5. Probability Calibration
            6. Calculating Performance
            7. Average AUPRC Calculation
            8. Statistical Analysis
        """
        print(f"\n=== Phase 2: Model Evaluation with {model_type.upper()} ===")

        if self.feature_importance_ranking is None:
            raise ValueError("Must run phase1_feature_selection first")

        all_results = []

        # For each feature count
        for n_features in feature_counts:
            print(f"\nEvaluating with {n_features} features...")

            # Step 1: Create feature subset
            # According to the feature importance ranking created in Phase 1,
            # subsets consisting of the first n features with the highest SHAP importance are determined.
            if n_features == 29:
                selected_features = self.feature_names
            else:
                selected_features = self.feature_importance_ranking[:n_features]

            X_subset = self.X[selected_features]

            # Step 2 & 3: Cross-validation with single class training
            # Ten repetitions (ten rounds) of five-fold cross-validation are used for model evaluation.
            # Since we are working with single class classifiers, after the data is divided into training and test subsets,
            # minority class (anomaly) examples in the training data are removed.
            # The class distribution in the test subset is not changed.
            fold_results = []

            # 10 repetitions of 5-fold CV
            for repeat in range(10):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
                print(f"  Processing repeat {repeat}/10...")

                for fold, (train_idx, test_idx) in enumerate(skf.split(X_subset, self.y)):
                    X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
                    y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                    # Step 3: Remove minority class from training data
                    normal_train_idx = y_train == 0
                    X_train_normal = X_train[normal_train_idx]

                    # Step 4: Train model
                    # Class SVM and One-Class GMM models are trained for each feature subset (3, 5, 7, 10, 15 and 29 features).
                    # The hyperparameters of the model are the same as in Phase 1.
                    if model_type == 'svm':
                        model = OneClassSVM(nu=0.0025, kernel='rbf', gamma=0.0310)
                    else:
                        model = OneClassGMM(n_components=2)

                    model.fit(X_train_normal)

                    # Step 5: Probability calibration: todo bu adımı bu şekilde yaptım ama hiç emin değilim bir daha bir gözden geçirsek iyi olur
                    # Create calibration data (using some normal + some anomaly samples)
                    # The output (decision scores) of the One-Class SVM and One-Class GMM models are subjected
                    # to sigmoid calibration to obtain the class probabilities required for the AUPRC metric.
                    # We did these because there were very few anomalous samples and logistic regression gives an error because there are no anomalous samples.
                    cal_size = min(1000, len(X_train))
                    # Since we use StratifiedKFold, we should have both classes
                    normal_indices = np.where(y_train == 0)[0]
                    anomaly_indices = np.where(y_train == 1)[0]

                    # Sample from both classes for calibration (maintain class balance)
                    n_normal_cal = min(cal_size // 2, len(normal_indices))
                    n_anomaly_cal = min(cal_size - n_normal_cal, len(anomaly_indices))

                    selected_normal = np.random.choice(normal_indices, n_normal_cal, replace=False)
                    selected_anomaly = np.random.choice(anomaly_indices, n_anomaly_cal, replace=False)

                    cal_idx = np.concatenate([selected_normal, selected_anomaly])
                    X_cal, y_cal = X_train.iloc[cal_idx], y_train.iloc[cal_idx]

                    # Convert to binary classification format for calibration
                    y_cal_binary = (y_cal == 1).astype(int)  # 1 for fraud, 0 for normal

                    # For calibration, we need to invert the decision function
                    # because OneClass models output higher scores for inliers
                    # One-Class models (e.g. OneClassSVM, GMM) produce higher scores for inliers (normal)
                    # samples and lower scores for anomaly samples.
                    # However, calibration algorithms (e.g. LogisticRegression) interpret these scores as "high score = high probability of anomaly".
                    # Therefore, the decision_function score is negated, thus inverting its meaning.
                    class CalibratedOneClass:
                        def __init__(self, model):
                            self.model = model

                        def decision_function(self, X):
                            return -self.model.decision_function(X)  # Invert scores

                        def predict(self, X):
                            return self.model.predict(X)

                    calibrated_model = CalibratedOneClass(model)

                    # Fit sigmoid calibrator
                    # cal_scores: Scores (decision_function output), rendered in 2D with .reshape(-1, 1) (LogisticRegression expects this).
                    # y_cal_binary: Real labels (0: normal, 1: anomaly)
                    # LogisticRegression(): Learns a sigmoid function and learns to convert decision_function values to probabilities.
                    cal_scores = calibrated_model.decision_function(X_cal).reshape(-1, 1)
                    calibrator = LogisticRegression(solver="lbfgs",
                                                    class_weight="balanced",
                                                    max_iter=1000)
                    calibrator.fit(cal_scores, y_cal_binary)

                    # Step 6: Calculate performance
                    # The performance of the model for each test fold is calculated and recorded using the
                    # Area Under the Precision Recall Curve (AUPRC) metric.
                    test_scores = calibrated_model.decision_function(X_test).reshape(-1, 1)
                    test_probs = calibrator.predict_proba(test_scores)[:, 1]

                    # Calculate AUPRC
                    precision, recall, _ = precision_recall_curve(y_test, test_probs)
                    auprc = auc(recall, precision)

                    fold_results.append({
                        'repeat': repeat,
                        'fold': fold,
                        'n_features': n_features,
                        'auprc': auprc
                    })

            all_results.extend(fold_results)

            # Step 7: Average AUPRC calculation
            # Since ten repeats of five-fold cross validation are performed, a total of 50 AUPRC scores are recorded for each model.
            # 50 scores are obtained for each number of features (6 different values), resulting in a total of 300 AUPRC scores for each classifier.
            # The average of these 50 scores is taken to obtain the average AUPRC performance for each number of features.
            avg_auprc = np.mean([r['auprc'] for r in fold_results])
            std_auprc = np.std([r['auprc'] for r in fold_results])

            print(f"Average AUPRC with {n_features} features: {avg_auprc:.4f} ± {std_auprc:.4f}")

        self.results[model_type] = pd.DataFrame(all_results)
        return self.results[model_type]

    def statistical_analysis(self, model_type='svm', alpha=0.01):
        """
        Step 8: Statistical analysis of results

        Statistical analysis is performed on the average AUPRC results. ANOVA (Analysis of Variance)
        test is used to determine the effect of the number of features on the experimental results.
         If the p-value as a result of the ANOVA test is below the predetermined significance level (0.01),
         it is concluded that the number of features has a significant effect. In this case,
         Tukey's Honestly Significant Difference (HSD) test is applied to rank the effects of different number
          of features on performance.
        The HSD test ranks the number of features into groups according to their effects on AUPRC scores.

        """
        # todo: bu fonksiyonun gözden geçirilmesi lazım
        print(f"\n=== Statistical Analysis for {model_type.upper()} ===")

        if model_type not in self.results:
            raise ValueError(f"No results found for {model_type}. Run phase2_model_evaluation first.")

        results_df = self.results[model_type]

        # Group by number of features
        grouped = results_df.groupby('n_features')['auprc']

        # Summary statistics
        summary = grouped.agg(['mean', 'std', 'count']).round(4)
        print("\nSummary Statistics:")
        print(summary)

        # ANOVA test
        groups = [group['auprc'].values for name, group in results_df.groupby('n_features')]
        f_stat, p_value = stats.f_oneway(*groups)

        print(f"\nANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant at alpha={alpha}: {'Yes' if p_value < alpha else 'No'}")

        # Tukey's HSD test if ANOVA is significant
        if p_value < alpha:
            print(f"\nTukey's HSD Test:")

            # Prepare data for Tukey test
            tukey_data = []
            tukey_groups = []

            for name, group in results_df.groupby('n_features'):
                tukey_data.extend(group['auprc'].values)
                tukey_groups.extend([f'{name}_features'] * len(group))

            tukey_result = pairwise_tukeyhsd(tukey_data, tukey_groups, alpha=alpha)
            print(tukey_result)

        return summary, f_stat, p_value

    def plot_results(self, model_types=['svm', 'gmm']):
        """Plot comparison of results"""
        fig, axes = plt.subplots(1, len(model_types), figsize=(6 * len(model_types), 5))
        if len(model_types) == 1:
            axes = [axes]

        for i, model_type in enumerate(model_types):
            if model_type in self.results:
                data = self.results[model_type]

                # Box plot
                sns.boxplot(data=data, x='n_features', y='auprc', ax=axes[i])
                axes[i].set_title(f'{model_type.upper()} - AUPRC by Number of Features')
                axes[i].set_xlabel('Number of Features')
                axes[i].set_ylabel('AUPRC')
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, file_path, model_types=['svm', 'gmm']):
        """Run the complete two-phase analysis"""
        print("Starting complete credit card fraud detection analysis...")

        # Load data
        self.load_and_prepare_data(file_path)

        # Run analysis for each model type
        for model_type in model_types:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING WITH {model_type.upper()} MODEL")
            print(f"{'=' * 60}")

            # Phase 1: Feature Selection
            self.phase1_feature_selection(model_type)

            # Phase 2: Model Evaluation
            self.phase2_model_evaluation(model_type)

            # Statistical Analysis
            self.statistical_analysis(model_type)

        # Plot results
        if len(model_types) > 0:
            self.plot_results(model_types)

        return self.results
