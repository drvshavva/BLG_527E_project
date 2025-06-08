from src.anomaly_detection.credit_card import CreditCardFraudDetection

# paper selected features
# gmm: 'V11', 'V2', 'V26', 'V12', 'V8', 'V5', 'V9', 'V7', 'V18', 'V3', 'V10', 'V16', 'V17', 'V21', 'V14'
# svm: 'V1', 'V23', 'V25', 'V3', 'V18', 'V28', 'V2', 'V19', 'V21', 'V15', 'V5', 'V20', 'V4', 'V11', 'V16'
if __name__ == "__main__":
    # Initialize the detector
    detector = CreditCardFraudDetection(
        _feature_importance_ranking=['V1', 'V23', 'V25', 'V3', 'V18', 'V28', 'V2', 'V19',
                                     'V21', 'V15', 'V5', 'V20', 'V4', 'V11', 'V16'])

    # Run complete analysis
    # Replace 'creditcard.csv' with your dataset path
    results = detector.run_complete_analysis(
        file_path='./dataset/creditcard.csv',
        model_types=['gmm', 'svm'],
        feature_counts=[3, 5, 7, 10, 15, 29]
    )

    # Access individual results
    print("\nFinal Results Summary:")
    for model_type in ['gmm', 'svm']:
        if model_type in results:
            df = results[model_type]
            summary = df.groupby('n_features')['auprc'].agg(['mean', 'std'])
            print(f"\n{model_type.upper()} Results:")
            print(summary.round(4))
