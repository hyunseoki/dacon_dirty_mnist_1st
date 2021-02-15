import pandas as pd
import glob
import datetime
import os
import argparse
from datetime import datetime


def ensemble(base_path) -> None:
    fn_list = glob.glob(os.path.join(base_path, '*.csv'))
    assert os.path.isdir(base_path)

    df = pd.read_csv(fn_list[0])
    assert len(fn_list) > 1

    for fn in fn_list[1:]:
        df.iloc[:,1:] += pd.read_csv(fn).iloc[:,1:]

    df.iloc[:,1:] = (df.iloc[:,1:] / len(fn_list) > 0.5).astype(int)
    df.to_csv(
        os.path.join(base_path, datetime.now().strftime("%m%d%H%M") + '_ensemble_submission.csv'),
        index=False
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="./올려야 되는 것들/")
    args = parser.parse_args()
    ensemble(args.base_path)


if __name__ == '__main__':
    main()