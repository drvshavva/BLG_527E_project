from src.anomaly_detection.anomaly_detection import CreditCardFraudDetection

if __name__ == "__main__":
    # Initialize the detector
    detector = CreditCardFraudDetection()

    # Run complete analysis
    # Replace 'creditcard.csv' with your dataset path
    results = detector.run_complete_analysis(
        file_path='./dataset/creditcard.csv',
        model_types=['svm', 'gmm']
    )

    # Access individual results
    print("\nFinal Results Summary:")
    for model_type in ['svm', 'gmm']:
        if model_type in results:
            df = results[model_type]
            summary = df.groupby('n_features')['auprc'].agg(['mean', 'std'])
            print(f"\n{model_type.upper()} Results:")
            print(summary.round(4))