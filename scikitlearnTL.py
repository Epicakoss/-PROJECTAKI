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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
# Streamlit app for interactive widgets

# --- Title of the App ---
st.title("Interactive Data Analysis and Modeling App")

#Add a Sidebar
st.sidebar.title("Visualization Options")
st.sidebar.write("Upload a dataset and explore various features")

global df
df = None
#Setup Sidebar file upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your data file", 
    type=["h5ad", "hdf5", "csv", "xlsx"], 
    label_visibility="collapsed"
)

# Initialize a global DataFrame
# Using 'df = None' is a simpler way to handle this
df = None
if uploaded_file is not None:
    print (uploaded_file)
    print("Hello, you uploaded a file!")
    try:
        # We'll primarily focus on CSV as per the model training code
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(f"Could not read as CSV, trying Excel. Error: {e}")
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e2:
            st.error(f"Failed to read the file. Error: {e2}")


# Display the DataFrame if it exists
global numeric_columns
try:
    # Display the DataFrame
    #st.write(df)
    numeric_columns = list (df.select_dtypes(['float','int']).columns)
except Exception as e:
    print (e)
    st.write("No data uploaded yet.")



# Add a multi-tab interface
if uploaded_file is not None:
    st.title("Multi-tab Interface")
    tab1, tab2, tab3 ,tab4, tab5,tab6= st.tabs(["Dataset", "Statistics and Clustering", "Plots", "Dynamic Filtering","Model Training","Predict on New Data"])
    with tab1:
        #Display the uploaded data
        st.write("### Uploaded Data")
        st.dataframe(df)
    with tab2:
        #Summary Statistics
        st.write("### Summary Statistics")
        st.write(df.describe())
        st.write("### Clustering Results")
        st.dataframe(df)
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
    with tab5:
       st.write("This tab uses the uploaded data to train a model.")
        
       # We add a button to prevent the model from retraining every time you interact
       # with another widget.
       if st.button("Train Model"):
            # It uses the 'df' that was already loaded when you uploaded the file.

            # We assume the last column is the target (y) and all others are features (X).
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            
            # Make predictions and calculate accuracy
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Display the result
            st.write(f"### Model Accuracy: {acc:.2f}")

    with tab6:
      st.write(" Upload the neccessary files to predict on new data.") 
      uploaded_model = st.file_uploader("Upload Trained Model", type=["pkl"])
      uploaded_test = st.file_uploader("Upload Test CSV", type=["csv"])

      if uploaded_model and uploaded_test:
        # Load trained model
        model = joblib.load(uploaded_model)

        # Load test dataset
        test_data = pd.read_csv(uploaded_test)

        # Identify the correct features used during training
        expected_features = model.feature_names_in_

        # Drop any extra columns that were not used during training
        test_data_filtered = test_data[expected_features]

        # Make predictions
        predictions = model.predict(test_data_filtered)

        # Add predictions to the dataframe
        test_data["Predicted Class"] = predictions

        # Display results
        st.write("### Predictions")
        st.dataframe(test_data)           

             

# Add a message to the user if no file has been uploaded yet.
if df is None:
    st.info("Awaiting for a file to be uploaded.")

# Sidebar settings  
st.sidebar.header("Settings")
if uploaded_file is not None:
        k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df.iloc[:, :-1])
        #The last two lines are commented out because it appears in the tabs section
        #st.write("### Clustering Results")
        #st.dataframe(df)

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