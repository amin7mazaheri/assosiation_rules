import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
df=pd.read_csv('LAX-air-pollution.csv')
X=df
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
orgexplained=p.explained_variance_ratio_.cumsum()[1]


explained=np.zeros(1000)
for i in np.arange(1000):
    s=np.random.choice(X.shape[0],X.shape[0],replace=True)
    Xnew=scaledX[s,:]
    p=PCA()
    p.fit(Xnew)
    explained[i]=p.explained_variance_ratio_.cumsum()[1]

plt.hist(explained,bins=60,color=(1,0.94,0.86),edgecolor=(0.54,0.51,0.47))
plt.axvline(np.quantile(explained,0.975),color=(0.46,0.93,0))
plt.axvline(np.quantile(explained,0.025),color=(0.46,0.93,0))
plt.axvline(orgexplained,color='red')
plt.show()
