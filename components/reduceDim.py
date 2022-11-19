import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, TSNE


def reduceDimension(args, X):

    X_fix = X[:2, :]
    X_variate = X[2:, :]

    X_variate = X_variate.transpose((1, 0))

    if args.method.lower() == 'tsne':
        tsne = TSNE(n_components=args.target_size, perplexity=args.perplexity, n_iter=args.n_iter,
                    init='pca', random_state=0, method='exact')
        X_variate = tsne.fit_transform(X_variate)

    elif args.method.lower() == 'pca':
        pca = PCA(n_components=args.target_size)
        pca.fit(X_variate)
        X_variate = pca.transform(X_variate)

    elif args.method.lower() == 'le':
        le = SpectralEmbedding(n_components=args.target_size)
        X_variate = le.fit_transform(X_variate)


    X_variate = X_variate.transpose((1, 0))
    X = np.concatenate((X_fix, X_variate), axis=0)

    return X
