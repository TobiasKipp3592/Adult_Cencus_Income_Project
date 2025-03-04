import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
import io
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from data_cleaning_app import fill_missing_values, rename_columns, clean_data
from fpdf import FPDF

# openai api
load_dotenv()
client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Title
st.title("AI Assistant for Data Science 🤖")

# Welcoming message
st.write("Hello! 👋 Welcome to the AI Assistant for Data Science. This tool is designed to help you perform Exploratory Data Analysis on your dataset and solve Data Science Problems. Simply upload your CSV file, and let our AI guide you through the insights. Get ready to uncover hidden patterns and make data-driven decisions!")

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

        
        llm = ChatOpenAI(model="gpt-4", temperature=0)

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

        if not st.session_state.cleaned:
            if st.button("Clean Dataset"):
                st.session_state.df_cleaned = clean_data(df)
                st.session_state.cleaned = True  
                st.rerun()

        if st.session_state.cleaned:
            st.success("✅ Your data has successfully been cleaned!")
            st.divider()

        if st.session_state.df_cleaned is not None:
            st.write("### Data after Cleaning")
            st.write(st.session_state.df_cleaned.head())
            st.write(st.session_state.df_cleaned.tail())
        else:
            st.write("Dataset has not been cleaned yet.")

        if st.session_state.cleaned:
            st.divider()
            st.header("Descriptive Data Analysis")

        questions = [
            "How is income (≤50K / >50K) distributed in the general population, and what patterns or anomalies can be identified?",
            "How does the probability of earning more than 50K change with increasing age?",
            "Are there significant income differences between men and women?",
            "How does income distribution (≤50K / >50K) differ among various ethnic groups?",
            "Is there a relationship between a person's country of origin and their income class (≤50K / >50K)?"
        ]

        if "remaining_questions" not in st.session_state:
            st.session_state.remaining_questions = questions.copy()
            st.session_state.completed_analyses = []

        if "df_cleaned" not in st.session_state or st.session_state.df_cleaned is None:
            st.error("No dataset uploaded. Please upload a dataset before proceeding.")
        else:
            if st.session_state.remaining_questions:
                selected_question = st.session_state.remaining_questions[0]
                agent = create_pandas_dataframe_agent(llm, st.session_state.df_cleaned, verbose=True, allow_dangerous_code=True)
                response = agent.run(selected_question)
                st.subheader("Analysis Result:")
                st.write(response)
                
                fig, ax = plt.subplots()
                if "income" in selected_question:
                    st.session_state.df_cleaned['income'] = st.session_state.df_cleaned['income'].astype(str)
                    df_grouped = st.session_state.df_cleaned.groupby(['income']).size()
                    df_grouped.plot(kind='bar', ax=ax)
                    ax.set_title("Income Distribution")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Income Category")
                
                elif "age" in selected_question:
                    df_grouped = st.session_state.df_cleaned.groupby(['age', 'income']).size().unstack()
                    df_grouped.plot(kind='line', ax=ax)
                    ax.set_title("Income Distribution by Age")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Age")
                
                elif "gender" in selected_question:
                    df_grouped = st.session_state.df_cleaned.groupby(['sex', 'income']).size().unstack()
                    df_grouped.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title("Income Distribution by Gender")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Gender")
                
                elif "ethnic" in selected_question:
                    df_grouped = st.session_state.df_cleaned.groupby(['race', 'income']).size().unstack()
                    df_grouped.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title("Income Distribution by Ethnic Group")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Ethnic Group")
                
                elif "country of origin" in selected_question:
                    df_grouped = st.session_state.df_cleaned.groupby(['native-country', 'income']).size().unstack()
                    df_grouped.plot(kind='barh', stacked=True, ax=ax)
                    ax.set_title("Income Distribution by Country of Origin")
                    ax.set_ylabel("Country of Origin")
                    ax.set_xlabel("Count")
                
                st.pyplot(fig)
                st.session_state.completed_analyses.append((selected_question, response, fig))
                
                if st.button("Continue with another question"):
                    st.session_state.remaining_questions.pop(0)
                    st.rerun()
                
                if not st.session_state.remaining_questions:
                    st.success("All questions have been analyzed!")
                    
                    if st.button("Save results as PDF"):
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, "Analysis Report", ln=True, align='C')
                        
                        for idx, (question, analysis, fig) in enumerate(st.session_state.completed_analyses):
                            pdf.add_page()
                            pdf.cell(0, 10, f"{idx+1}. {question}", ln=True)
                            pdf.multi_cell(0, 10, analysis)
                            
                            image_path = f"temp_chart_{idx}.png"
                            fig.savefig(image_path)
                            pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=180)
                            os.remove(image_path)
                        
                        pdf_output_path = "analysis_report.pdf"
                        pdf.output(pdf_output_path)
                        with open(pdf_output_path, "rb") as pdf_file:
                            st.download_button(label="Download PDF Report", data=pdf_file, file_name="analysis_report.pdf", mime="application/pdf")