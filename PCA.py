import numpy as np

class PCA:
    def __init__(self,n_compoents):
        assert n_compoents>=1,'n_compoents must be vaild'
        self.n_compoents=n_compoents
        self.components_=None

    def fit(self,X,eta=0.01,n_iters=1e4):
        assert self.n_compoents<=X.shape[1],\
            'n_compoents must be greater than the feature of X'

        def demean(X):
            return X-np.mean(X,axis=0)

        def f(w,X):
            return np.sum((X.dot(w)**2))/len(X)

        def df(w,X):
            return X.T.dot(X.dot(w))*2./len(X)

        def direction(w):
            return w/np.linalg.norm(w)

        def first_compoents(X,initial_w,eta=0.01,n_iters=1e4,epsilon=1e-8):
            w=direction(initial_w)
            cur_iter=0
            while cur_iter<n_iters:
                gradient=df(w,X)
                last_w=w
                w=direction(w)
                if(abs(f(w,X)-f(last_w,X))<epsilon):
                    break
                cur_iter+=1
            return w

        X_pca=demean(X)
        self.components_=np.empty(shape=(self.n_compoents,X.shape[1]))
        for i in range(self.n_compoents):
            initial_w=np.random.random(X_pca.shape[1])
            w=first_compoents(X_pca,initial_w,eta,n_iters)
            self.components_[i,:]=w
            X_pca=X_pca-X_pca.dot(w).reshape(-1,1)*w
        return self
    def transform(self,X):
        assert X.shape[1]==self.components_.shape[1]
        return X.dot(self.components_.T)
    def inverse_transform(self,X):
        assert X.shape[1]==self.components_.shape[0]
        return X.dot(self.components_)
    def __repr__(self):
        return 'PCA(n_compoents=%d)'%self.n_compoents