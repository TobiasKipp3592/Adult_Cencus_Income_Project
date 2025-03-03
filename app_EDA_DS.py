import os 
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import io
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
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

        # Functions main
        
       # def cleaning_agent():
       #    st.write("Do you want to clean the data?")


        # Main

        st.divider()
        st.header("Data Science Problem")
        st.write("Now that we have digged deeper into our data, let's define a data science problem.")

        prompt = st.text_input("Add your science problem here")

        data_problem_template = PromptTemplate(
            input_variables=["data_science_problem"],
            template= "Convert the following problem into a data science problem: {data_science_problem}",
            )
        
        model_selection_template = PromptTemplate(
            input_variables=["data_problem"],
            template= "Give a list of ten machine learning algorithms that are suitable to solve this problem: {data_problem}",
            )
        
        data_problem_chain = data_problem_template | llm

        def format_data_problem(inputs):
            return {"data_problem": inputs["data_problem"]} 
        
        model_selection_chain = model_selection_template | llm
        
        sequential_chain = RunnableMap({
            "data_problem": data_problem_chain
        }) | RunnableLambda(format_data_problem) | RunnableMap({
            "data_problem": lambda x: x["data_problem"],  # Pass through the data_problem
            "model_selection": model_selection_chain
        })



        if prompt:
            response = sequential_chain.invoke({"data_science_problem": prompt})
            st.subheader("Data Science Problem:")
            st.write(response["data_problem"])
            st.subheader("Suggested Machine Learning Models:")
            st.write(response["model_selection"])
            
        
