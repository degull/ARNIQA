import pickle
import numpy as np
from sklearn.linear_model import Ridge
from pathlib import Path
from scipy import stats

def train_ridge_regressor(features, scores, alpha, output_path):
    """
    Ridge Regressor 학습 및 저장
    """
    # 모델 학습
    regressor = Ridge(alpha=alpha)
    regressor.fit(features, scores)

    # 모델 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(regressor, f)
    print(f"Regressor saved to {output_path}")

    return regressor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True, help="Path to features (.npy file)")
    parser.add_argument("--scores_path", type=str, required=True, help="Path to scores (.npy file)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for Ridge regression")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save trained regressor")
    args = parser.parse_args()

    # 데이터 로드
    features = np.load(args.features_path)
    scores = np.load(args.scores_path)

    # 모델 학습 및 저장
    train_ridge_regressor(features, scores, args.alpha, args.output_path)
