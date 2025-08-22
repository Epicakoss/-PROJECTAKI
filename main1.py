import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import hdf5plugin
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scanpy.external as sce
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Streamlit app for interactive widgets

#Title of the app
st.title("Welcome to our Application")

#Add a Sidebar
st.sidebar.title("Visualization Options")

#Setup Sidebar file upload
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["h5ad", "hdf5", "csv", "xlsx"], label_visibility="collapsed")

# Initialize a global DataFrame
global df
if uploaded_file is not None:
    print (uploaded_file)
    print("Hello, you uploaded a file!")
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print("Error reading file")
        df = pd.read_excel(uploaded_file)

# Display the DataFrame if it exists
global numeric_columns
try:
    # Display the DataFrame
    #st.write(df)
    numeric_columns = list (df.select_dtypes(['float','int']).columns)
except Exception as e:
    print (e)
    st.write("No data uploaded yet.")


# Streamlit app for interactive widgets
# Add a multi-tab interface
if uploaded_file is not None:
    st.title("Multi-tab Interface")
    tab1, tab2, tab3 ,tab4= st.tabs(["Dataset", "Statistics", "Plots", "Dynamic Filtering"])
    with tab1:
        #Display the uploaded data
        st.write("### Uploaded Data")
        st.dataframe(df)
    with tab2:
        #Summary Statistics
        st.write("### Summary Statistics")
        st.write(df.describe())
    with tab3:
        #Plotting options
        x_axis = st.selectbox("X-axis", df.columns[:-1])
        y_axis = st.selectbox("Y-axis", df.columns[:-1])
        fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[-1])
        st.plotly_chart(fig)
    with tab4:
        search_term = st.text_input("Filter Species:")
        filtered_df = df[df[df.columns[-1]].astype(str).str.contains(search_term, case=False, na=False)]
        st.dataframe(filtered_df)
        
st.sidebar.header("Settings")
if uploaded_file is not None:
        k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df.iloc[:, :-1])
        st.write("### Clustering Results")
        st.dataframe(df)

#Slider for Opinions
opinion_slider=st.select_slider("Pick opinion", options=["bad","mid","peak"])
st.write("Your opinion is ", opinion_slider)

# Color Picker
color = st.color_picker("Pick A Color", "#00f900")
st.write("The current color is", color)

# Movie Genre Selection
genre = st.radio(
    "What's your favorite movie genre",
    [":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"],
    index=None,
)
st.write("Your favorite movie genre:", genre)

#Uploading an Image
image_file = st.file_uploader("Upload image", type=["jpg", "png", "csv", "xlsx"], label_visibility="collapsed" )
#Image display
if image_file is not None:
    st.image(image=image_file, width=300, caption="Beautiful image", use_container_width=True)

#Insert audio
st.audio("Nintendo Wii - Mii Channel Theme.mp3", format="audio/mpeg")

# Options for data processing
options=["Jellybeans", "Fish Biscuit", "Madam President"]
#Multiple selection bar
multiple = st.multiselect("What are your favorite cat names?",
    options=options,
)
# Display selected options
st.write("You selected:", multiple)







