import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
df=pd.read_csv('./LAX-air-pollution.csv')
X=df
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
W=p.components_.T
y=p.fit_transform(scaledX)
yhat=X.dot(W)
plt.figure(1)
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="blue",marker='o',alpha=0.5)
plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

names=df.index
#'''
for i, txt in enumerate(names):
    plt.annotate('day{}'.format(txt), (yhat.iloc[i,0]*1.02, yhat.iloc[i,1]*1.02))

pd.DataFrame(W[:,:3],index=df.columns,columns=['PC1','PC2','PC3'])
pd.DataFrame(p.explained_variance_ratio_,index=np.arange(7)+1,columns=['Explained Variability'])
plt.figure(2)
plt.bar(np.arange(1,8),p.explained_variance_,color="blue",edgecolor="red")
plt.xlabel('Number of Components')
plt.ylabel('Explained Variability')


xs=yhat.iloc[:,0]
ys=yhat.iloc[:,1]
plt.figure(3)
for i in range(len(W[:,0])):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(np.mean(xs), np.mean(ys), W[i,0]*max(xs), -W[i,1]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(W[i,0]*max(xs)+np.mean(xs), +np.mean(ys)-W[i,1]*max(ys),
             list(df.columns.values)[i], color='r')
#'''
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="blue",marker='o',alpha=0.5)

for i in range(len(xs)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'bo')
    plt.text(xs[i]*1.025, ys[i]*1.025, list(df.index)[i], color='b')
plt.xlim(min(xs)-5,max(xs)+20)
#plt.ylim(min(score[:,1])-.9,max(score[:,1])+.9)
plt.xlabel('PC Scores1')
plt.ylabel('PC Scores2')
#plt.tight_layout()
plt.show()
#'''

#'''
