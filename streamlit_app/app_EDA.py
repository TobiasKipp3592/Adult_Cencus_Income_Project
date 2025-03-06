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
from datetime import datetime

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

            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, dict) and "content" in response:
                return response["content"].strip()
            else:
                return str(response)

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
            with st.expander("What is EDA?"):
                st.markdown(steps_eda())

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

        if st.session_state.cleaned:
            st.divider()
            st.header("📊 Descriptive Data Analysis")
            st.write("""Descriptive data analysis involves summarizing and visualizing the main features of a dataset. 
                     It helps in understanding the distribution, central tendency, and variability of the data. 
                     By examining these characteristics, we can uncover patterns, trends, and potential anomalies that inform further analysis and decision-making.""")

            questions = [
                "How is income (≤50K / >50K) distributed in the general population, and what patterns or anomalies can be identified?",
                "How does the probability of earning more than 50K change with increasing age?",
                "Are there significant income differences in gender between men and women?",
                "How does income distribution (≤50K / >50K) differ among various ethnic groups?",
                "Is there a relationship between a person's country of origin and their income class (≤50K / >50K)?"
            ]

            if "selected_questions" not in st.session_state:
                st.session_state.selected_questions = []
            if "current_question_index" not in st.session_state:
                st.session_state.current_question_index = 0
            if "completed_analyses" not in st.session_state:
                st.session_state.completed_analyses = []

            st.write("Select the questions you want to analyze:")
            selected_questions = []
            for i, question in enumerate(questions):
                if st.checkbox(question, key=f"q_{i}"):
                    selected_questions.append((i, question))

            if st.button("Start Analysis"):
                if not selected_questions:
                    st.warning("Please select at least one question before starting the analysis.")
                else:
                    st.session_state.selected_questions = selected_questions
                    st.session_state.current_question_index = 0
                    st.session_state.completed_analyses = []
                    st.rerun()

            @st.cache_data
            def data_analysis(question_index, question, df_cleaned, _llm):
                agent = create_pandas_dataframe_agent(llm, df_cleaned, verbose=True, allow_dangerous_code=True)
                response = agent.run(question)
                
                fig, ax = plt.subplots()
                plt.style.use("dark_background")
                colors = ["silver", "teal"]

                if question_index == 0:
                    df_cleaned['income'] = df_cleaned['income'].astype(str)
                    df_grouped = df_cleaned.groupby(['income']).size()
                    df_grouped.plot(kind='bar', ax=ax, color=colors)
                    ax.set_title("Income Distribution")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Income Category")

                elif question_index == 1:
                    df_grouped = df_cleaned.groupby(['age', 'income']).size().unstack()
                    df_grouped.plot(kind='line', ax=ax, color=colors)
                    ax.set_title("Income Distribution by Age")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Age")

                elif question_index == 2:
                    df_grouped = df_cleaned.groupby(['sex', 'income']).size().unstack()
                    df_grouped.plot(kind='bar', stacked=True, ax=ax, color=colors)
                    ax.set_title("Income Distribution by Gender")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Gender")

                elif question_index == 3:
                    df_grouped = df_cleaned.groupby(['race', 'income']).size().unstack()
                    df_grouped.plot(kind='bar', stacked=True, ax=ax, color=colors)
                    ax.set_title("Income Distribution by Ethnic Group")
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Ethnic Group")

                elif question_index == 4:
                    df_grouped = df_cleaned.groupby(['native_country', 'income']).size().unstack()

                    top_countries = df_cleaned['native_country'].value_counts().nlargest(10).index
                    df_grouped = df_grouped.loc[top_countries]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    df_grouped.plot(kind='barh', stacked=True, ax=ax, color=["gray", "teal"])
                    ax.set_title("Top 10 Income Distribution by Country of Origin")
                    ax.set_ylabel("Country of Origin")
                    ax.set_xlabel("Count")
                    ax.legend(title="Income", labels=["<=50K", ">50K"])

                return response, fig

            if "df_cleaned" in st.session_state and st.session_state.df_cleaned is not None:
                for prev_question, prev_response, prev_fig in st.session_state.completed_analyses:
                    st.subheader(f"📊 {prev_question}") 
                    st.write(prev_response)
                    st.pyplot(prev_fig)

                if st.session_state.selected_questions:
                    index = st.session_state.current_question_index
                    if index < len(st.session_state.selected_questions):
                        question_index, question = st.session_state.selected_questions[index]

                        if not any(q == question for q, _, _ in st.session_state.completed_analyses):
                            st.subheader(f"📊 {question}")
                            response, fig = data_analysis(question_index, question, st.session_state.df_cleaned, llm)

                            st.session_state.completed_analyses.append((question, response, fig))
                            st.rerun()

                        if st.button("Continue with next question"):
                            st.session_state.current_question_index += 1
                            st.rerun()
                    else:
                        st.success("All selected questions have been analyzed!")

            class CustomPDF(FPDF):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.font_path = r"C:\Users\tobia\PortfolioProject\fonts\DejaVuSans.ttf"
                    self.add_font("DejaVu", "", self.font_path, uni=True)
                    self.set_font("DejaVu", "", 12)

                def header(self):
                    if self.page > 1:
                        self.set_font("DejaVu", "", 10)
                        self.cell(0, 10, f"Data Analysis Report", align="C", border=1)
                        self.ln(15)

                def footer(self):
                    if self.page > 1:
                        self.set_y(-15)
                        self.set_font("DejaVu", "", 10)
                        self.cell(0, 10, f"Page {self.page-1}", align="R")

            def create_professional_report(completed_analyses):
                pdf = CustomPDF()
                pdf.set_auto_page_break(auto=True, margin=15)

                pdf.add_page()
                pdf.set_font("DejaVu", "", 24)
                pdf.cell(0, 20, "Data Analysis Report", ln=True, align='C')
       
                logo_path = r"C:\Users\tobia\PortfolioProject\assets\3515462.jpg"
                if os.path.exists(logo_path):
                    pdf.image(logo_path, x=50, y=80, w=100)

                pdf.ln(170)
                pdf.set_font("DejaVu", "", 16)
                pdf.cell(0, 10, "Adult Income Dataset", ln=True, align='C')
                
                pdf.ln(20)
                pdf.set_font("DejaVu", "", 12)
                pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%d.%m.%Y at %H:%M Uhr')}", ln=True, align='C')
                pdf.cell(0, 10, f"Author: Tobias Kipp", ln=True, align='C')

                section_titles = [
                    "Income Distribution",
                    "Income Distribution by Age", 
                    "Income Distribution by Gender",
                    "Income Distribution by Ethnic Group",
                    "Top 10 Income Distribution by Country of Origin"
                ]

                for idx, (question, analysis, fig) in enumerate(completed_analyses):
                    pdf.add_page()
                    
                    pdf.set_font("DejaVu", "", 16)
                    pdf.set_x(10)
                    pdf.cell(0, 10, section_titles[idx], ln=True, align='L', border="B")
                    pdf.ln(2)
                    
                    pdf.set_font("DejaVu", "", 12)
                    pdf.set_x(10)
                    pdf.multi_cell(0, 10, f"Analytics questions: {question}")
                    pdf.ln(5)
                    
                    pdf.set_font("DejaVu", "", 11)
                    pdf.multi_cell(0, 7, analysis)
                    pdf.ln(10)
                    
                    image_path = f"temp_chart_{idx}.png"
                    fig.savefig(image_path)
                    pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=180)
                    os.remove(image_path)

                pdf_bytes = pdf.output(dest="S").encode("latin1")
                return io.BytesIO(pdf_bytes)

            if st.button("Generate PDF-Report"):
                if not hasattr(st.session_state, 'completed_analyses') or not st.session_state.completed_analyses:
                    st.error("Bitte führen Sie zunächst Analysen durch.")
                    st.stop()

                pdf_buffer = create_professional_report(st.session_state.completed_analyses)
                
                st.success("✅ PDF-Report has been successfully generated!")

                st.download_button(
                    label="📥 Download PDF-Report",
                    data=pdf_buffer,
                    file_name="data_analytics_report.pdf",
                    mime="application/pdf"
                )