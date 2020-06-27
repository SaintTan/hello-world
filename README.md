# python package instruction

This is the python instruction and codes for data-mining, data-preprocessing, and machine learning, while using package like numpy, pandas, scikit-learn, matplotlib, latex.

## numpy
***
> + import numpy as np
>
> + linalg
>
>   ```python
>   import numpy.linalg as LA
>   LA.norm(W)
>   LA.svd(A)
>   ```
>
>   
***
## pandas
***
> + import pandas as pd
> + how to get multi-index element
```python
## train_set is a DataFrame and the columns is [drug1,drug2,synergistic score]
arrays = [train_set[0].values.tolist(),train_set[1].values.tolist()]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['drug1', 'drug2'])
train_E_set = pd.DataFrame(np.zeros((len(index),k1+k2)), index=index)
## train_E_set.loc[idx_drug1,idx_drug2]
## or train_E_set.loc[idx_drug1].loc[idx_drug2]
```
***
## scikit-learn
***
> + from sklearn.decomposition import TruncatedSVD
>
> + preprocessing
>
>   + normalization
>
>   ```python
>   from sklearn import preprocessing
>   W1 = preprocessing.normalize(W1, norm='l2')
>   ```
>
>   + train_test dataset split
>
>   ```python
>   from sklearn.model_selection import train_test_split
>   train, test = train_test_split(edgelist, test_size=0.1, random_state=42)
>   ```
>
> + Decomposition
>
>   + TruncatedSVD
>
>   ```python
>   from sklearn.decomposition import TruncatedSVD
>   svd = TruncatedSVD(n_components=k2, n_iter=10)
>   W = svd.fit_transform(A)
>   H = svd.components_.T
>   ```
>
>   + NMF
>
>   ```python
>   from sklearn.decomposition import NMF
>   model = NMF(n_components=k1, init='nndsvd', random_state=0, max_iter=600)
>   W = model.fit_transform(df_A1.values)
>   H = model.components_.T
>   ```
>
>   + PCA
>
>   ```python
>   from sklearn.decomposition import PCA
>   pca = PCA(n_components=2)
>   Draw_E = pca.fit_transform(np_edgelist_E_DrugComb_val)
>   ```
>
>   + KernelPCA
>
>   ```python
>   from sklearn.decomposition import KernelPCA
>   kpca = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True, gamma=10)
>   Draw_E = kpca.fit_transform(edgelist_E_DrugComb.values)
>   ```
>
>   + TSNE
>
>   ```python
>   from sklearn.manifold import TSNE
>   Draw_E = TSNE(n_components=2).fit_transform(edgelist_E_DrugComb.values)
>   ```
>
> + Kernel and Stochastic Gradient descent
>
>   + kernel_approximation
>
>   ```python
>   from sklearn.kernel_approximation import Nystroem
>   feature_map_nystroem = Nystroem(gamma=.2,random_state=1,n_components=400) # rbf kernel
>   data_transformed = feature_map_nystroem.fit_transform(np_edgelist)
>   ```
>
>   + Stocastic gradient classfication
>
>   ```python
>   from sklearn.linear_model import SGDClassifier
>   clf = SGDClassifier(loss="hinge", penalty="l2",alpha=0.001,class_weight="balanced",max_iter=500)
>   clf.fit(data_transformed, y)
>   ```
>
>   
>
> + Metric
>
>   + roc_auc_score
>
>   ```python
>   from sklearn.metrics import roc_auc_score
>   roc_auc_score(y_true, y_scores)
>   ```
>
>   
>
> + How to save models of scikit-learn
>
>   ```python
>   ## joblib is sklearn externals module
>   from sklearn.externals import joblib #joblib模块
>   joblib.dump(clf, './clf.pkl') # save the clf model
>   clf_ex = joblib.load('clf.pkl') # load the clf model
>   ```
>
>   

***
## networkx
***
> + import networkx as nx
***
## matplotlib
***
> + import matplotlib.pyplot as plt
***
## latex
***
> + \usepackage{graphicx}
