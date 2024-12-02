import pickle
import numpy as np
from pathlib import Path
from scipy import stats

def load_regressor(model_path):
    """
    저장된 Regressor 로드
    """
    with open(model_path, "rb") as f:
        regressor = pickle.load(f)
    print(f"Regressor loaded from {model_path}")
    return regressor

def predict_with_regressor(regressor, test_features, test_scores=None):
    """
    Ridge Regressor로 예측 수행 및 평가
    """
    preds = regressor.predict(test_features)

    if test_scores is not None:
        # SROCC 및 PLCC 계산
        srocc, _ = stats.spearmanr(preds, test_scores)
        plcc, _ = stats.pearsonr(preds, test_scores)
        print(f"SROCC: {srocc:.4f}, PLCC: {plcc:.4f}")
    else:
        print("No test scores provided. Only predictions will be returned.")
    
    return preds

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved regressor (.pkl file)")
    parser.add_argument("--test_features_path", type=str, required=True, help="Path to test features (.npy file)")
    parser.add_argument("--test_scores_path", type=str, default=None, help="Path to test scores (.npy file)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions (.npy file)")
    args = parser.parse_args()

    # 데이터 로드
    test_features = np.load(args.test_features_path)
    test_scores = np.load(args.test_scores_path) if args.test_scores_path else None

    # 모델 로드
    regressor = load_regressor(args.model_path)

    # 예측 수행
    preds = predict_with_regressor(regressor, test_features, test_scores)

    # 예측 결과 저장
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, preds)
    print(f"Predictions saved to {output_path}")
