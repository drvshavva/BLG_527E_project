import pandas as pd

from src.anomaly_detection.credit_card import CreditCardFraudDetection


def test():
    detector = CreditCardFraudDetection()

    detector.load_and_prepare_data('../dataset/creditcard.csv')
    assert isinstance(detector.data, pd.DataFrame)
    ranking, feature_importance_df = detector.phase1_feature_selection('gmm')
    assert isinstance(feature_importance_df, pd.DataFrame)
