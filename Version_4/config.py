import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ML_Model Example")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--fit_intercept", type=bool, default=False)
    return parser.parse_args()