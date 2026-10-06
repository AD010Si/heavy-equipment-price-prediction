"""Smoke test: full train -> predict run on synthetic data with tiny models."""
from src.predict import make_submission
from src.train import run
from tests.synthetic import make_frame


def test_train_and_predict(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    make_frame(1500, seed=1).to_csv(data_dir / "train.csv", index=False)
    make_frame(300, seed=2, with_target=False).to_csv(data_dir / "test.csv", index=False)

    metrics = run(data_dir, tmp_path / "models", tmp_path / "reports", fast=True)
    assert 0 < metrics["blend_rmsle_second_half_holdout"] < 1.5

    sub = make_submission(data_dir, tmp_path / "models", tmp_path / "sub.csv")
    assert len(sub) == 300 and (sub["TargetValue"] >= 0).all()
