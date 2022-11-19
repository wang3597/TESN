import argparse


def esn_argparse():
    parser = argparse.ArgumentParser()

    # Folders Information
    parser.add_argument('--data_path', default='./datasets', type=str)

    # DataSet Information
    parser.add_argument('--dataset', default='Sunspots', choices=['Lorenz', 'Sunspots'], type=str)
    parser.add_argument('--train_length', default=0, type=int)
    parser.add_argument('--test_length', default=0, type=int)
    parser.add_argument('--init_length', default=0, type=int)

    parser.add_argument('--in_size', default=1, type=int)
    parser.add_argument('--out_size', default=1, type=int)
    parser.add_argument('--reservoir_size', default=500, type=int)
    parser.add_argument('--spectral_radius', default=0.9, type=float)
    parser.add_argument('--sparse_degree', default=0.02, type=float)
    parser.add_argument('--input_scale', default=0.1, type=float)
    parser.add_argument('--leaking_rate', default=1, type=float)

    # Dimensionality Reduction Methods Information
    parser.add_argument('--useRD', default=False, type=bool)
    parser.add_argument('--method', default='tsne', type=str)
    parser.add_argument('--target_size', default=2, type=int)

    # TSNE
    parser.add_argument('--perplexity', default=30, type=int)
    parser.add_argument('--n_iter', default=300, type=int)
    parser.add_argument('--degrees_of_freedom', default=1, type=int)

    # Training Information
    parser.add_argument('--seed', default=42, type=int)

    return parser
