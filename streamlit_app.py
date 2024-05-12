import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


st.title('Iris Flower Detection')
sepal_length = st.slider("How old are you?", 0, 130, 25)
sepal_width = st.slider("How old are you?", 0, 10, 25)
petal_length = st.slider("How old are you?", 0, 30, 25)
petal_width = st.slider("How old are you?", 0, 0, 25)

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df['target'] = load_iris().target
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_classifier.score(X_test, y_test)
prediction = svm_classifier.predict(pd.DataFrame(np.array([[sepal_length, sepal_width, petal_length, petal_width
                                                            ]])))[0]

st.write(prediction)