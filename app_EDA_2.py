import os 
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import io
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from data_cleaning_app import fill_missing_values, rename_columns, clean_data

# openai api
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Title
st.title("AI Assistant for Data Science 🤖")

# Welcoming message
st.write("Hello! 👋 Welcome to the AI Assistant for Data Science. This tool is designed to help you perform Exploratory Data Analysis on your dataset. Simply upload your CSV file, and let our AI guide you through the insights. Get ready to uncover hidden patterns and make data-driven decisions!")

# Explanation sidebar
with st.sidebar:
    
    st.write("*Your Data Science Adventure Starts Here!*")
    st.caption("""**Here, you can effortlessly upload your CSV files and begin your data exploration journey.
    Our tool is designed to help you perform comprehensive Exploratory Data Analysis with ease.
    Simply upload your dataset, and let our AI guide you through the insights.
    Get ready to uncover hidden patterns and make data-driven decisions!
    You can upload your CSV file here.**""")

    st.divider()

    st.caption("<p style='text-align:center'>Made By Tobias Kipp</p>", unsafe_allow_html=True)

# Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True
st.button("Let's get started", on_click= clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your CSV file here", type=["csv"])
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False, na_values=["?"])

        # llm model
        llm = OpenAI(temperature=0)

        # Function sidebar
        @st.cache_data
        def steps_eda():
            prompt = "Explain the standard steps of Exploratory Data Analysis (EDA) with examples."
            response = llm(prompt)
            return response

        # Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write(df.tail())

            st.write("This is the information about your dataset:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

            st.write("**Identifying Unusual Unique Values**")
            unusual_values = {}
            for col in df.columns:
                value_counts = df[col].value_counts(dropna=False).to_frame().reset_index()
                value_counts.columns = ["Value", "Count"]
                unusual_values[col] = value_counts
            
            for col, table in unusual_values.items():
                st.write(f"**Column: {col}**")
                st.dataframe(table)
           
            st.write("**Missing Values**")
            missing_values = df.isnull().sum()
            st.write(missing_values)

            st.write("**Distinct and Duplicate Value Pairs**")
            duplicate_rows = df[df.duplicated(keep=False)]
            st.write(duplicate_rows)

        # Main

        st.header("Exploratory Data Analysis")
        st.subheader("General information about the dataset")

        with st.sidebar:
            with st.expander("What are the steps of EDA"):
                st.write(steps_eda())

        function_agent()

        st.header("**Would you like to clean the data?**")

        if "df_cleaned" not in st.session_state:
            st.session_state.df_cleaned = None
        if "cleaned" not in st.session_state:
            st.session_state.cleaned = False

        if st.button("Clean Dataset"):
            st.session_state.df_cleaned = clean_data(df)
            st.session_state.cleaned = True  
        
        if st.session_state.df_cleaned is not None:
            st.write("### Data after Cleaning")
            st.write(st.session_state.df_cleaned.head())
            st.write(st.session_state.df_cleaned.tail())
        else:
            st.write("Dataset has not been cleaned yet.")

        if st.session_state.cleaned:
            st.divider()
            st.header("Descriptive Data Analysis")
            st.subheader("Based on current information the following questions can be answered:")

            if st.checkbox("Wie verteilt sich das Einkommen (≤50K / >50K) in der Gesamtbevölkerung des *Adult Income Dataset*, und welche Muster oder Auffälligkeiten lassen sich erkennen?s"):
                st.write("Hier könnte die Antwort aus dem Prompt stehen")

            if st.checkbox("Wie verändert sich die Wahrscheinlichkeit, ein Einkommen von über 50K zu verdienen, mit zunehmendem Alter?"):
                st.write("Hier könnte die Antwort aus dem Prompt stehen")

            if st.checkbox("Gibt es signifikante Unterschiede im Einkommen zwischen Männern und Frauen im *Adult Income Dataset"):
                st.write("Hier könnte die Antwort aus dem Prompt stehen")

            if st.checkbox("Wie unterscheidet sich die Einkommensverteilung (≤50K / >50K) zwischen verschiedenen ethnischen Gruppen im *Adult Income Dataset"):
                st.write("Hier könnte die Antwort aus dem Prompt stehen")

            if st.checkbox("Gibt es einen Zusammenhang zwischen dem Herkunftsland einer Person und ihrer Einkommensklasse (≤50K / >50K)?"):
                st.write("Hier könnte die Antwort aus dem Prompt stehen")