import argparse
import os
import _init_paths
from lib.train.admin import create_default_local_file_ITP_train
from lib.test.evaluation import create_default_local_file_ITP_test


def parse_args():
    parser = argparse.ArgumentParser(description='Create default local file on ITP or PAI')
    parser.add_argument("--workspace_dir", type=str, required=True)  # workspace dir
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    workspace_dir = os.path.realpath(args.workspace_dir)
    data_dir = os.path.realpath(args.data_dir)
    save_dir = os.path.realpath(args.save_dir)
    create_default_local_file_ITP_train(workspace_dir, data_dir)
    create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir)
