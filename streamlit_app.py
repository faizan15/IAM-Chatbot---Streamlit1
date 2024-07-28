import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import plotly.express as px

# Load environment variables
load_dotenv()
env_var = load_dotenv()

# Set up OpenAI
openai_api_key = env_var['OPENAI_API_KEY']
client = OpenAI(api_key=openai_api_key)

# Set up Pinecone
pinecone_api_key = env_var['PINECONE_API_KEY']
pinecone_environment = env_var['PINECONE_ENVIRONMENT']

pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to a Pinecone index
index_name = 'iam-chatbot'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_environment
        )
    )
index = pc.Index(index_name)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('incidents.xlsx')
    df['combined_text'] = df['Class'] + " " + df['Description']
    return df

# Embed text using OpenAI
def embed_text(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

# Function to find similar incidents
def find_similar_incidents(new_incident_description):
    new_embedding = embed_text(new_incident_description)
    new_embedding_list = np.array(new_embedding).tolist()
    query_response = index.query(vector=new_embedding_list, top_k=3, include_values=True)
    similar_incidents = query_response['matches']
    return [int(match['id']) for match in similar_incidents]

# Function to suggest resolutions based on similar incidents
def suggest_resolutions(new_incident_description):
    similar_incident_ids = find_similar_incidents(new_incident_description)
    similar_resolutions = df.loc[similar_incident_ids, 'Resolution'].tolist()
    return similar_resolutions

# Streamlit UI
st.set_page_config(page_title="Incident Resolution AI", layout="wide")

st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        color: #FF9633;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Incident Resolution AI</p>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    df = load_data()
st.success("Data loaded successfully!")

# Sidebar
st.sidebar.header("Filters")
incident_class = st.sidebar.multiselect("Filter by Incident Class", options=df['Class'].unique())
if incident_class:
    df = df[df['Class'].isin(incident_class)]

# Assuming 'Date' column exists, add date range selector
if 'Date' in df.columns:
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [df['Date'].min(), df['Date'].max()]
    )
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Main content
view = st.radio("Select View", ["Chat", "Data"])

if view == "Chat":
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Describe the incident..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            suggested_resolutions = suggest_resolutions(prompt)
        
        response = "Here are some suggested resolutions:\n\n" + "\n\n".join(f"- {res}" for res in suggested_resolutions)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.button("Celebrate Resolution"):
            st.balloons()

else:
    st.subheader("Incident Data")
    st.dataframe(df)

    st.subheader("Incident Distribution")
    fig = px.pie(df, names='Class', title='Incidents by Class')
    st.plotly_chart(fig)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Incidents", len(df))
col2.metric("Unique Classes", df['Class'].nunique())
if 'Resolution Time' in df.columns:
    col3.metric("Avg. Resolution Time", f"{df['Resolution Time'].mean():.2f} hours")

# File uploader
uploaded_file = st.file_uploader("Upload custom incident file", type="xlsx")
if uploaded_file is not None:
    new_df = pd.read_excel(uploaded_file)
    st.write("New data uploaded:")
    st.dataframe(new_df)

    # Option to replace existing data
    if st.button("Replace existing data with uploaded data"):
        df = new_df
        st.success("Data replaced successfully!")
        st.experimental_rerun()

# Embed all combined texts (only if not already embedded)
if 'embeddings' not in df.columns:
    with st.spinner("Generating embeddings... This may take a while."):
        df['embeddings'] = df['combined_text'].apply(embed_text)
        vectors = [(str(i), np.array(emb).tolist()) for i, emb in enumerate(df['embeddings'].tolist())]
        index.upsert(vectors)
    st.success("Embeddings generated and stored in Pinecone!")

# Add information about the app
st.sidebar.title("About")
st.sidebar.info("This is an Incident Resolution Chat application. It uses AI to suggest resolutions based on similar past incidents.")
