import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title('streamlit example')
st.write('''
   # Explore different classifier \n
    which one is the best''')
datasetname = st.sidebar.selectbox('Select dataset',('Iris','Breast cancer','Wine dataset'))
# st.write(datasetname)
classifiername = st.sidebar.selectbox('Select classifier',('KNN','SVM','Random Forest'))

def get_dataset(datasetname):
    if datasetname == 'Iris':
        data = datasets.load_iris()
    elif datasetname == 'Breast cancer':
            data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
        
    X = data.data
    y = data.target
    
    return X,y

X,y = get_dataset(datasetname)
st.write('Shape of feature', X.shape)
st.write('Unique classes',len(np.unique(y)))

# add parameters

def add_parameter_ui(clf_name):
    params = {}
    if clf_name == 'KNN':
        K = st.sidebar.slider('k',1,15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params['C'] = C
    else:
        pass
        # max_depth = st.sidebar.slider('max_depth',2,15)
        # params['max_depth'] = max_depth
        # n_estimators = st.sidebar.slider('n_estimators',1,100)
        # params['n_estimators'] = n_estimators
        
    return params

params = add_parameter_ui(classifiername)

def get_classifier(clf_name,params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVC':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier()
                                       
            
    return clf


clf = get_classifier(classifiername, params)
        
    
# classification
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
cf = confusion_matrix(y_test,y_pred)
st.write(f"classifier ={classifiername}")
st.write(f"accuracy score ={acc}")
st.write(cf)

# plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig, ax = plt.subplots()
ax.scatter(x1, x2, c=y)

st.pyplot(fig)
