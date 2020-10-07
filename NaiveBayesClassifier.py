"""
朴素贝叶斯分类器（属性条件独立性假设）
"""
import numpy as np
from utils.dataset import dataSet
# 只考虑离散值
class NaiveBayesClassifier:
    def __init__(self,n_classes=2):
        self.n_classes=n_classes
        self.priori_P={}
        self.conditional_P={}
        self.N={}
        pass

    def fit(self,X,y):
        for i in range(self.n_classes):
            # 公式 7.19
            self.priori_P[i]=(len(y[y==i])+1)/(len(y)+self.n_classes)
        for col in range(X.shape[1]):
            self.N[col]=len(np.unique(X[:,col]))
            self.conditional_P[col]={}
            for row in range(X.shape[0]):
                val=X[row,col]
                if val not in self.conditional_P[col].keys():
                    self.conditional_P[col][val]={}
                    for i in range(self.n_classes):
                        D_xi=np.where(X[:,col]==val)
                        D_c=np.where(y==i)
                        D_cxi=len(np.intersect1d(D_xi,D_c))
                        # 公式 7.20
                        self.conditional_P[col][val][i]=(D_cxi+1)/(len(y[y==i])+self.N[col])
                else:
                    continue

    def predict(self,X):
        pred_y=[]
        for i in range(len(X)):
            p=np.ones((self.n_classes,))
            for j in range(self.n_classes):
                p[j]=self.priori_P[j]
            for col in range(X.shape[1]):
                val=X[i,col]
                for j in range(self.n_classes):
                    p[j]*=self.conditional_P[col][val][j]
            pred_y.append(np.argmax(p))
        return np.array(pred_y)

# 连续值
class NaiveBayesClassifierContinuous:
    def __init__(self,n_classes=2):
        self.n_classes=n_classes
        self.priori_P={}

    def fit(self,X,y):
        self.mus=np.zeros((self.n_classes,X.shape[1]))
        self.sigmas=np.zeros((self.n_classes,X.shape[1]))

        for c in range(self.n_classes):
            # 公式 7.19
            self.priori_P[c]=(len(y[y==c]))/(len(y))
            X_c=X[np.where(y==c)]

            self.mus[c]=np.mean(X_c,axis=0)
            self.sigmas[c]=np.std(X_c,axis=0)

    def predict(self,X):
        pred_y=[]
        for i in range(len(X)):
            p=np.ones((self.n_classes,))
            for c in range(self.n_classes):
                p[c]=self.priori_P[c]
                for col in range(X.shape[1]):
                    x=X[i,col]
                    p[c]*=1./(np.sqrt(2*np.pi)*self.sigmas[c,col])*np.exp(-(x-self.mus[c,col])**2/(2*self.sigmas[c,col]**2))
            pred_y.append(np.argmax(p))
        return np.array(pred_y)

if __name__=='__main__':
    X, Y, X_test = dataSet()
    naive_bayes=NaiveBayesClassifier(n_classes=2)
    naive_bayes.fit(X,Y)
    print('self.PrirP:',naive_bayes.priori_P)
    print('self.CondiP:',naive_bayes.conditional_P)
    pred_y=naive_bayes.predict(X_test)
    print('pred_y:',pred_y)


