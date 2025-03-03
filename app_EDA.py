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
        df = pd.read_csv(user_csv, low_memory=False)

        # llm model
        llm = OpenAI(temperature=0)

        # Function sidebar
        @st.cache_data
        def steps_eda():
            prompt = "Explain the standard steps of Exploratory Data Analysis (EDA) with examples."
            response = llm(prompt)
            return response

        # Pandas Agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

        # Clean the data
        df_cleaned = clean_data(df)

        # Pandas Agent cleaned Data
        pandas_agent_cleaned = create_pandas_dataframe_agent(llm, df_cleaned, verbose=True, allow_dangerous_code=True)

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

            st.write("**Data Cleaning**")
            st.write("Unique values per column:")
            unique_values = {col: df[col].nunique() for col in df.columns}
            st.write(unique_values)

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

            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return
        
        # def cleaning_agent():
        #     df_cleaned = clean_data(df)
        #     return df_cleaned

        @st.cache_data
        def function_question_variable():
            if df[user_question_variable].dtype in ['int64', 'float64']:
                st.line_chart(df, y=[user_question_variable])
            else:
                st.bar_chart(df[user_question_variable].value_counts())

            summary_statistics = pandas_agent.run(
                f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question_variable}"
            )
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return 

        # Main

        st.header("Exploratory Data Analysis")
        st.subheader("General information about the dataset")

        with st.sidebar:
            with st.expander("What are the steps of EDA"):
                st.write(steps_eda())

        function_agent()
        # st.write("Based on given informationen I suggest this steps for a cleaned dataset: Fill missing values and rename columns")
        # st.checkbox("Clean the dataset", on_change=cleaning_agent)
        # cleaning_agent()
        # st.write("The dataset is now cleaned")

        st.subheader("Variable of study")

        user_question_variable = st.text_input("What variable are you interested in")
        if user_question_variable is not None and user_question_variable != "":
            function_question_variable()

        # This Function need more flexibilty: The Chart should be choosen from which dtype the variable is
        
            st.subheader("Further study")

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                    function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                    st.write("")

            if user_question_dataframe:
                st.divider()
                st.header("Descriptive Data Analysis")
                st.write("Now that we have a better understanding of the data, let's begin with descriptive data analysis. We will ask questions about our dataset to uncover its main characteristics.")

                prompt = st.text_input("Add your prompt here")

                data_analysis_template = PromptTemplate(
                    input_variables=["data_analysis_question"],
                    template="Analyze the following dataset and provide insights with clear bullet points: {data_analysis_question}",
                    )

                data_analysis_chain = data_analysis_template | llm

                if prompt:
                    response = data_analysis_chain.invoke({"data_analysis_question": prompt})
                    st.write(response)

                    if prompt:
                        st.divider()
                        st.header("Data Science Problem")
                        st.write("Now that we have digged deeper into our data, let's define a data science problem.")
                        
                        prompt = st.text_input("Add your business problem here")

                        data_problem_template = PromptTemplate(
                            input_variables=["data_science_problem"],
                            template="Define a data science problem based on the following business problem: {data_science_problem}",
                            )

                        data_problem_chain = data_problem_template | llm
                        if prompt:
                            response = data_problem_chain.invoke({"data_science_problem": prompt})
                            st.write(response)