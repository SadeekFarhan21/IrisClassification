import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

st.set_page_config(
    page_title="Iris Flower Detection",
    page_icon="🌸",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Set title and page configuration
st.title('Iris Flower Detection')
st.sidebar.header('User Input')

# Sliders for user input
sepal_length = st.sidebar.slider('Sepal Length', min_value=4.0, max_value=8.0, step=0.1, value=5.0)
sepal_width = st.sidebar.slider('Sepal Width', min_value=1.5, max_value=4.5, step=0.1, value=3.0)
petal_length = st.sidebar.slider('Petal Length', min_value=0.5, max_value=7.0, step=0.1, value=1.0)
petal_width = st.sidebar.slider('Petal Width', min_value=0.0, max_value=3.0, step=0.1, value=0.5)

# EDA
st.header('Exploratory Data Analysis (EDA)')
st.subheader('Iris Dataset Overview:')
st.write(pd.DataFrame(X, columns=iris.feature_names).describe())

# Pairplot
st.subheader('Pairplot of Features:')
pairplot_fig = px.scatter_matrix(pd.DataFrame(X, columns=iris.feature_names), dimensions=iris.feature_names, color=y,
                                 title="Pairplot of Features", height=900, width=900)
st.plotly_chart(pairplot_fig)
# Code snippet for Pairplot
st.subheader('Code Snippet for Pairplot:')
st.code("""
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
pairplot_fig = px.scatter_matrix(pd.DataFrame(iris.data, columns=iris.feature_names), dimensions=iris.feature_names, color=iris.target)
pairplot_fig.show()
""", language='python')

# Correlation Heatmap
st.subheader('Correlation Heatmap:')
corr_matrix = pd.DataFrame(X, columns=iris.feature_names).corr()
fig = px.imshow(corr_matrix, x=iris.feature_names, y=iris.feature_names, color_continuous_scale='blues',
                title="Correlation Heatmap")

# Add text annotations
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        fig.add_annotation(
            x=iris.feature_names[i],
            y=iris.feature_names[j],
            text=str(round(corr_matrix.iloc[i, j], 2)),
            showarrow=False,
            font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        )

st.plotly_chart(fig)
# Code snippet for Correlation Heatmap
st.subheader('Code Snippet for Correlation Heatmap:')
st.code("""
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
corr_matrix = pd.DataFrame(iris.data, columns=iris.feature_names).corr()
fig = px.imshow(corr_matrix, x=iris.feature_names, y=iris.feature_names, color_continuous_scale='blues')
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        fig.add_annotation(
            x=iris.feature_names[i],
            y=iris.feature_names[j],
            text=str(round(corr_matrix.iloc[i, j], 2)),
            showarrow=False,
            font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        )

fig.show()
""", language='python')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Support Vector Classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict the class for the user input
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = svm_classifier.predict(user_input)[0]

# Display prediction
st.header('Prediction')
st.subheader('Predicted Iris Species:')
st.write(iris.target_names[prediction].capitalize())

# Additional information
st.write('---')
st.subheader('About Iris Dataset:')
st.write('The Iris dataset contains measurements of sepal and petal lengths and widths for 150 iris flowers, '
         'divided into three species: Setosa, Versicolor, and Virginica.')
st.write('---')
st.subheader('Reference:')
st.write('This Streamlit app is for demonstration purposes and uses the Iris dataset for classification.')
