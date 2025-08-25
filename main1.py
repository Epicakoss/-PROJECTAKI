# Necessary imports for your application
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.express as px 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image
# Streamlit app for interactive widgets

#Load and display an image
image = Image.open('experiment icon image.jpg')

#Config function
st.set_page_config(
    page_title="Data Analysis and Modeling App",
    page_icon=image
)

# --- Title of the App ---
st.title(":test_tube: Interactive Data Analysis and Modeling App")

#Add a Sidebar
st.sidebar.title(":loudspeaker: Visualization Options")
st.sidebar.write(":minidisc: Upload a dataset and explore various features")

global df
df = None
#Setup Sidebar file upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your data file", 
    type=["h5ad", "hdf5", "csv", "xlsx"],
    label_visibility="collapsed"
)

# Initialize a global DataFrame
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
    # Display the DataFrame if it exists
    numeric_columns = list (df.select_dtypes(['float','int']).columns)
except Exception as e:
    print (e)
    st.write("No data uploaded yet.")


# Streamlit app for interactive widgets
# Add a multi-tab interface
if uploaded_file is not None:
    st.title("Multi-tab Interface")
    tab1, tab2, tab3 ,tab4,tab5,tab6, tab7= st.tabs(["Dataset", "Statistics and Clustering", "Plots", "Dynamic Filtering","Model Training","Predict on New Data", "About us!"])
    with tab1:
        #Display the uploaded data
        st.write("### Uploaded Data")
        st.dataframe(df)
    with tab2:
        with st.expander(":arrow_double_down: Summary Statistics"):
            with st.container(border=True):
                #Summary Statistics and Clustering
                st.write("### Show Summary Statistics")
                st.write(df.describe())
        with st.expander(":arrow_double_down: Clustering Results"):
            with st.container(border=True):
                #Sidebar for clustering
                st.sidebar.header(":bulb: Settings")
                if uploaded_file is not None:
                    k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    df['Cluster'] = kmeans.fit_predict(df.iloc[:, :-1])
                    #The last two lines are commented out because it appears in the tabs section
                    st.write("### Show Clustering Results")
                    st.dataframe(df)
    with tab3:
        #Plotting options
        st.write("### Scatter Plot")
        x_axis = st.selectbox(":chart: X-axis", df.columns[:-1])
        y_axis = st.selectbox(":chart: Y-axis", df.columns[:-1])
        fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[-1])
        st.plotly_chart(fig)
    with tab4:
        search_term = st.text_input(":dart: Filter Species:")
        filtered_df = df[df[df.columns[-1]].astype(str).str.contains(search_term, case=False, na=False)]
        st.dataframe(filtered_df)
    with tab5:
        st.write(":jigsaw: This tab uses the uploaded data to train a model.")
        # We add a button to prevent the model from retraining every time you interact
        # with another widget.
        if st.button("Train Model"):

            # Assume the last column is the target
            y = df.iloc[:, -1]

            # A simple way is to just take all numeric columns as features
            X = df[numeric_columns]

            # If the target column is in numeric_columns, drop it from X
            if y.name in X.columns:
                X = X.drop(columns=[y.name])
            # Check if X is empty after dropping the target column
            if X.empty:
                st.error("No valid feature columns found for training. Please check your data.")
            else:
                # Create and train the model

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
      #Predict on New Data
      st.write(" Upload the neccessary files to predict on new data.") 
      uploaded_model = st.file_uploader(":jigsaw: Upload Trained Model", type=["pkl"],key="model_uploader")#ADDED KEY
      uploaded_test = st.file_uploader(":jigsaw: Upload Test CSV", type=["csv"],key="test_data_uploader")#ADDED KEY

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
    with tab7:
        #About Us
        st.write("""
        ## About Us
        This application was developed by a team of informatics to facilitate interactive data analysis and modeling. 
        Our goal is to provide users with an intuitive platform to explore datasets, visualize trends, and build predictive models with ease.
        
        ### Team Members Roles:
        - Ζιώγας Χρυσοβαλάντης(inf2022053): Data Analyst, Machine Learning Engineer, Frontend Developer.
        - Παυσανίας Κόντος(inf2022083): Dockerization, Report Latex.

        ### Our Work:
        Ζιώγας Χρυσοβαλάντης(inf2022053):
        - Developed an interactive web application using Streamlit.
        - Implemented data visualization features with Plotly.
        - Integrated machine learning capabilities using Scikit-learn.
        - Designed a user-friendly interface with multiple tabs for different functionalities.
        - Designed and implemented the user interface and overall application architecture.
        
        
        Παυσανίας Κόντος(inf2022083):
        - Assisted in the development and testing of the application(Machine learning studying and coding).
        - Containerized the application using Docker for easy deployment.
        - Created comprehensive documentation and report using LaTeX.
        - Emotional Support during the development process. :P <3
        - "Nah I'd win"
        """)
    
        #Insert audio
        st.write("### Enjoy some music while you explore!")
        st.audio("Nintendo Wii - Mii Channel Theme.mp3", format="audio/mpeg")

# Add a message to the user if no file has been uploaded yet.
if df is None:
    st.info("Awaiting for a file to be uploaded.")









