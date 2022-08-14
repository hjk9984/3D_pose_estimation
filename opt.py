import argparse

def base_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config")
    parser.add_argument("--is_train", action='store_true')
    parser.add_argument("--data_path", default="/Users/antae/Dev_hj/data/")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_validate", action="store_true")

    args = parser.parse_args()
    return args
