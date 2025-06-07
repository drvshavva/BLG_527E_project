import pandas as pd

from src.anomaly_detection.anomaly_detection import CreditCardFraudDetection


def test():
    detector = CreditCardFraudDetection()

    detector.load_and_prepare_data('../dataset/creditcard.csv')
    assert isinstance(detector.data, pd.DataFrame)
    _, feature_importance_df = detector.phase1_feature_selection('gmm')
    assert isinstance(feature_importance_df, pd.DataFrame)
