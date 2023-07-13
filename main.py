# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st


#favicon
st.set_page_config(page_title="Mall Customer Clustering", page_icon="🧊")

#style.css

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# import dataset

df = pd.read_csv('dataset/Mall_Customers.csv')

# rename columns
df.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'}, inplace=True)

# drop unneeded columns
X = df.drop(['CustomerID','Gender'], axis=1)

st.title('Mall Customers Clustering')
st.write(X)

st.write('---')

#create clusters with elbow method

st.subheader('Elbow Method')

clusters = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    clusters.append(kmeans.inertia_)

#create visualization

fig,ax = plt.subplots(figsize=(10,5))
sns.lineplot(x=range(1,11), y=clusters, ax=ax)
ax.set_title('Elbow Method')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')

# set annotate
ax.annotate('Elbow Method', xy=(5,clusters[4]), xytext=(3,clusters[4]*1.1), arrowprops=dict(facecolor='blue'))

st.pyplot(fig)

st.sidebar.subheader('Select Number of Clusters')
clust = st.sidebar.slider('Number of Clusters', 1, 10, 3,1)

# create function to run kmeans
def run_kmeans(n_clust):
    kmeans = KMeans(n_clusters=n_clust, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)

    X['label'] = kmeans.labels_

# create visualization
    fig,ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(x=X['Annual_Income'], y=X['Spending_Score'], hue=X['label'], ax=ax,palette=sns.color_palette('hls',n_clust))
    ax.set_title('Clusters')
    ax.set_xlabel('Annual Income')
    ax.set_ylabel('Spending Score')

    for i, txt in enumerate(X['label']):
        ax.annotate(txt, (X['Annual_Income'][i], X['Spending_Score'][i]))

    st.write('---')

    st.pyplot(fig)
    st.write(X)

run_kmeans(clust)

st.write('---')
#footer
st.markdown('''
     - All rights reserved - @masendhy - 2023 -
    ''', unsafe_allow_html=True)