#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize, scale

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


data =pd.read_excel('ResilienceAnalysi.xlsx')
data.columns = ['name']+[ i for i in range(1,8)]
kvcc = data.loc[2].values
data = data.drop(2)
data.reset_index(drop=True, inplace=True)
data.iloc[:,1:] = data.iloc[:,1:].astype(float)
data.iloc[:,1:]  = data.iloc[:,1:].apply(lambda x: round(x,2))
names = data['name']
mean= data.mean(axis=1)
data = data.drop('name', axis=1)

#%%%
from sklearn.decomposition import PCA

standardscaler = StandardScaler()
data = standardscaler.fit_transform(data)
pca = PCA(n_components=2)
# pca = PCA(n_components=3)

pca.fit(data)
X_pca = pca.transform(data)
df =pd.DataFrame(X_pca)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
px.line(sse, title='elbow curve', labels={'value':'inertia', 'index': 'number of clusters'})


# %%

kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
# dbscan = DBSCAN(eps=0.3)
# dbscan.fit(df)
# %%
newdf= df.copy()
df['mean']= mean
df.columns = ['pca1', 'pca2', 'mean']
# df.columns = ['pca1', 'pca2','pca3']

df['names'] = names
df['c'] = kmeans.labels_
df['c']= df['c'].astype(str)
# %%
df['c'] = df['c'].astype(str)


fig=px.scatter(df, x='pca1', y='pca2',color='c', hover_name='mean',text='names')

fig.update_traces(textposition='top center')

fig.show()
# %%

scaled_all4= standardscaler.transform(np.array(7*[4]).reshape(1,-1))
all_4 = pca.transform(scaled_all4)
kmeans.predict(all_4)

# %%
scaled_all3= standardscaler.transform(np.array(7*[3]).reshape(1,-1))
all_3 = pca.transform(scaled_all3)
kmeans.predict(all_3)

# %%
scaled_all2= standardscaler.transform(np.array(7*[2]).reshape(1,-1))
all_2 = pca.transform(scaled_all2)
kmeans.predict(all_2)
# %%
scaled_all1= standardscaler.transform(np.array(7*[1]).reshape(1,-1))
all_1 = pca.transform(scaled_all1)
kmeans.predict(all_1)
# %%
scaled_all0= standardscaler.transform(np.array(7*[0]).reshape(1,-1))
all_0 = pca.transform(scaled_all0)
kmeans.predict(all_0)

#%%%%%%%%%%%%
outlier = standardscaler.transform((kvcc[1:].reshape(1,-1)))
out= pca.transform(outlier)
                                                       
#%%%%%%%
df.loc[df.shape[0]+1]= [all_4[0][0], all_4[0][1]] + [4] + ['all4'] + [kmeans.predict(all_4)[0]]

#%%%%%%%%
df.loc[df.shape[0]+1]= [all_3[0][0], all_3[0][1]] + [3] + ['all3'] + [kmeans.predict(all_3)[0]]

# %%
df.loc[df.shape[0]+1]= [all_2[0][0], all_2[0][1]] + [2] + ['all2'] + [kmeans.predict(all_2)[0]]
# %%
df.loc[df.shape[0]+1]= [all_1[0][0], all_1[0][1]] + [1] + ['all1'] + [kmeans.predict(all_1)[0]]

# %%
df.loc[df.shape[0]+1]= [all_0[0][0], all_0[0][1]] + [0] + ['all0'] + [kmeans.predict(all_0)[0]]

# %%
df.loc[df.shape[0]+1]= [out[0][0], out[0][1]] + [1.45] + ['KVCC_outlier'] + [kmeans.predict(out)[0]]

# %%
df['c'] = df['c'].astype(str)
# fig=px.scatter_3d(df, x='pca1', y='pca2',z='mean' ,
#                   color='c', hover_name=df['names'],
#                   symbol=df['c'].astype(str),size='mean')

fig=px.scatter(df, x='pca1', y='pca2',color='c', 
               hover_name='mean',text='names',
               color_discrete_sequence=["red", "yellow", "green"])

fig.update_traces(textposition='top center')
# %%
fig.write_html("with_allClasses.html")
# %%

# %%
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'])
loading_matrix
# %%

# %%
# %%
df.to_csv('data.csv')
# %%




