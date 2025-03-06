import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import re
import traceback
import matplotlib.pyplot as plt
from IPython.display import display
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
import io
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from streamlit_app.data_cleaning_app import fill_missing_values, rename_columns, clean_data
from fpdf import FPDF
import tempfile

# openai api
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    st.caption("<p style='text-align:center'>Made By Tobias Kipp & Kimberly Koblinsky</p>", unsafe_allow_html=True)

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

        
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        

        # Function sidebar
        @st.cache_data
        def steps_eda():
            prompt = "Explain the standard steps of Exploratory Data Analysis (EDA) with examples."
            response = llm(prompt)
            return response

        @st.cache_data
        def data_science_framing():
            data_science_framing = llm("Write a couple of paragraphs about the importance of framing a data science problem approriately")
            return data_science_framing

        @st.cache_data
        def algortihm_selection():
            data_science_framing = llm("Write a couple of paragraphs about the importance of choosing the right algorithm and of considering more than one algorithm when trying to solve a data science problem.")
            return data_science_framing


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

        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=["data_science_problem"],
                template= "Convert the following problem into a concise data science problem in one sentence: {data_science_problem}",
            )
        
            model_selection_template = PromptTemplate(
                input_variables=["data_problem"],
                template= "Give a list of 10 machine learning algorithms that are suitable to solve this problem: {data_problem}",
            )
            return data_problem_template, model_selection_template
        
        @st.cache_resource
        def chains():
            data_problem_template, model_selection_template = prompt_templates()
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
            return sequential_chain
        
        @st.cache_data
        def chains_output(prompt):
            my_chain = chains()
            my_chain_output = my_chain.invoke({"data_science_problem": prompt})
            my_data_problem = my_chain_output["data_problem"]
            _my_model_selection = my_chain_output["model_selection"]
            return my_data_problem, _my_model_selection
        
        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            algorithm_lines = my_model_selection_input.split("\n")
            algorithms = [algorithm.split(":")[-1].split(".")[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
            algorithms.insert(0, "Select Algorithm")
            formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
            return formatted_list_output

        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm = llm,
                tool = PythonREPLTool(),
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                max_iterations = 5,
            )
            return agent_executor
        
        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, df):
            # Instead of relying on the agent to generate full code,
            # we'll create a template and use the LLM directly
            
            # Get column names and some sample data
            columns = df.columns.tolist()
            sample_data = df.head(2).to_string()
            data_types = df.dtypes.to_string()
            
            # Prepare a prompt for the LLM
            prompt = f"""
            You are a Python expert specializing in data science and machine learning.
            
            TASK: Create a comprehensive Python script to solve this problem: {my_data_problem}
            
            ALGORITHM TO USE: {selected_algorithm}
            
            DATASET INFORMATION:
            - Shape: {df.shape[0]} rows and {df.shape[1]} columns
            - Columns: {', '.join(columns)}
            - Data types: 
            {data_types}
            
            Sample data:
            {sample_data}
            
            Your code should:
            1. Import all necessary libraries (pandas, numpy, sklearn, matplotlib, etc.)
            2. Include proper dataset loading: df = pd.read_csv('dataset.csv')
            3. Perform necessary preprocessing (encoding categorical variables, handling missing values, scaling features)
            4. Split data into training and testing sets
            5. Implement and train the {selected_algorithm} model
            6. Make predictions and evaluate using standard metrics (accuracy, precision, recall, F1-score)
            7. Include fairness analysis with fairlearn (EOD, DPD, DPO)
            8. Visualize results with matplotlib or seaborn
            9. Visualize results of the fainress analysis with matplotlib or seaborn
            10. Include detailed comments explaining each step
            
            Return ONLY Python code, no explanations before or after. The code should be complete and ready to run.
            """
            
            # Use the LLM directly instead of the agent
            solution = llm.invoke(prompt).content
            
            # Clean up the response to ensure it's valid Python code
            if "```python" in solution:
                solution = solution.split("```python")[1].split("```")[0].strip()
            elif "```" in solution:
                solution = solution.split("```")[1].split("```")[0].strip()
                
            return solution
        
        def execute_python_code(code_string, dataframe):
            """
            Execute the generated Python code safely and return the results and errors.
            This version modifies the code to handle file path references.
            
            Args:
                code_string (str): The Python code to execute
                dataframe (pd.DataFrame): The DataFrame to make available to the code
                
            Returns:
                tuple: (output_string, error_string, figures, metrics_dict)
            """
            # Create output capture
            stdout_capture = io.StringIO()
            
            # Store original stdout and capture figures
            original_stdout = sys.stdout
            original_show = plt.show
            figures = []
            metrics = {}
            
            # Function to capture figures instead of displaying them
            def capture_figure(*args, **kwargs):
                fig = plt.gcf()
                figures.append(fig)
                plt.close(fig)
            
            try:
                # Redirect stdout
                sys.stdout = stdout_capture
                print("Starting code execution...")
                
                # Replace plt.show with our capture function
                plt.show = capture_figure
            
                # Find patterns like: df = pd.read_csv('file.csv') or data = pd.read_csv("file.csv")
                modified_code = code_string
                
                # Common patterns for data loading
                csv_patterns = [
                    r'(\w+)\s*=\s*pd\.read_csv\([\'"].*?[\'"](,.*?)?\)',
                    r'(\w+)\s*=\s*pandas\.read_csv\([\'"].*?[\'"](,.*?)?\)'
                ]
                
                # Replace file reading with direct dataframe assignment
                for pattern in csv_patterns:
                    matches = re.findall(pattern, modified_code)
                    for match in matches:
                        var_name = match[0] 
                        old_code = re.search(f"{var_name}\\s*=\\s*p[d|andas]\\.read_csv\\(['\"].*?['\"].*?\\)", modified_code).group(0)
                        print(f"Replacing file loading: {old_code}")
                        new_code = f"{var_name} = df.copy()"
                        modified_code = modified_code.replace(old_code, new_code)
                
                print("Modified code to use provided dataframe instead of loading from file")
                
                # Prepare variable namespace for code execution
                namespace = {
                    'df': dataframe,
                    'pd': pd,
                    'np': np,
                    'plt': plt,
                    'metrics': metrics
                }
                
                # Execute the modified code
                print("Executing modified code...")
                exec(modified_code, namespace)
                print("Code execution completed.")
                
                # Get any metrics that were collected
                if 'metrics' in namespace:
                    metrics.update(namespace['metrics'])
                
                # No errors if we get here
                error_output = None
                
            except Exception as e:
                # Capture any exceptions
                error_output = traceback.format_exc()
                print(f"Error during execution: {str(e)}")
            finally:
                # Restore original stdout and plt.show
                output = stdout_capture.getvalue()
                sys.stdout = original_stdout
                plt.show = original_show
            
            # Return all results
            return output, error_output, figures, metrics

        def display_execution_results(output, error, figures, metrics):
            """
            Display the results of code execution in Streamlit.
            
            Args:
                output (str): Console output from code execution
                error (str): Error messages if any
                figures (list): List of matplotlib figures
                metrics (dict): Dictionary of collected metrics
            """
            # Display success message
            if not error:
                st.success("Code executed successfully!")
            
            # Display console output
            st.subheader("Console Output")
            st.text_area("", output, height=200)
            
            # Display any errors
            if error:
                st.subheader("Errors")
                st.error(error)
            
            # Display figures
            if figures:
                st.subheader(f"Generated Charts ({len(figures)})")
                for i, fig in enumerate(figures):
                    st.pyplot(fig)
            else:
                st.warning("No charts were generated during execution.")
            
            # Display metrics if available
            if metrics:
                st.subheader("Model Metrics")
                st.json(metrics)

        def interpret_fairness_metrics(metrics_dict):
            """
            Interpretiert die Fairness-Metriken und zeigt sie in Streamlit an.
            
            Args:
                metrics_dict (dict): Ein Dictionary mit berechneten Metriken
            """
            st.subheader("Fairness-Metriken Interpretation")
            
            if not metrics_dict:
                st.warning("Keine Metriken zur Interpretation gefunden.")
                return
            
            # Modellleistungs-Metriken anzeigen
            st.write("### Modellleistung")
            cols = st.columns(4)
            
            if "accuracy" in metrics_dict:
                cols[0].metric("Genauigkeit", f"{metrics_dict['accuracy']:.4f}")
            if "precision" in metrics_dict:
                cols[1].metric("Präzision", f"{metrics_dict['precision']:.4f}")
            if "recall" in metrics_dict:
                cols[2].metric("Recall", f"{metrics_dict['recall']:.4f}")
            if "f1" in metrics_dict:
                cols[3].metric("F1-Score", f"{metrics_dict['f1']:.4f}")
            
            # Fairness-Metriken anzeigen
            if "demographic_parity" in metrics_dict or "equalized_odds" in metrics_dict:
                st.write("### Fairness-Metriken")
                
                fcols = st.columns(2)
                
                if "demographic_parity" in metrics_dict:
                    dpd = metrics_dict["demographic_parity"]
                    fcols[0].metric("Demographic Parity Difference", f"{dpd:.4f}")
                    
                    if abs(dpd) < 0.05:
                        fcols[0].success("Faire Vorhersageverteilung über alle demografischen Gruppen.")
                    elif abs(dpd) < 0.1:
                        fcols[0].warning("Leichte Unterschiede in der Vorhersageverteilung.")
                    else:
                        fcols[0].error("Deutliche Unterschiede in der Vorhersageverteilung.")
                        
                if "equalized_odds" in metrics_dict:
                    eod = metrics_dict["equalized_odds"]
                    fcols[1].metric("Equalized Odds Difference", f"{eod:.4f}")
                    
                    if abs(eod) < 0.05:
                        fcols[1].success("Nahezu gleiche Chancen für alle Gruppen.")
                    elif abs(eod) < 0.1:
                        fcols[1].warning("Leichte Ungleichheit in den Chancen.")
                    else:
                        fcols[1].error("Signifikante Ungleichheit in den Chancen.")
            
                # Trade-off zwischen Genauigkeit und Fairness
                st.write("### Trade-off: Genauigkeit vs. Fairness")
                
                accuracy = metrics_dict.get("accuracy", 0)
                dpd = metrics_dict.get("demographic_parity", 1)
                eod = metrics_dict.get("equalized_odds", 1)
                max_fairness_gap = max(abs(dpd) if dpd is not None else 0, abs(eod) if eod is not None else 0)
                
                if accuracy > 0.8 and max_fairness_gap < 0.1:
                    st.success("Dieses Modell bietet eine gute Balance zwischen Genauigkeit und Fairness.")
                elif accuracy > 0.8:
                    st.warning("Hohe Genauigkeit, aber mit Fairness-Bedenken. Erwägen Sie Fairness-Constraints.")
                elif max_fairness_gap < 0.1:
                    st.warning("Gute Fairness, aber die Genauigkeit könnte verbessert werden.")
                else:
                    st.error("Sowohl Genauigkeit als auch Fairness sind verbesserungswürdig.")



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

        # questions = [
        #     "How is income (≤50K / >50K) distributed in the general population, and what patterns or anomalies can be identified?",
        #     "How does the probability of earning more than 50K change with increasing age?",
        #     "Are there significant income differences between men and women?",
        #     "How does income distribution (≤50K / >50K) differ among various ethnic groups?",
        #     "Is there a relationship between a person's country of origin and their income class (≤50K / >50K)?"
        # ]

        # if "remaining_questions" not in st.session_state:
        #     st.session_state.remaining_questions = questions.copy()
        #     st.session_state.completed_analyses = []

        # if "df_cleaned" not in st.session_state or st.session_state.df_cleaned is None:
        #     st.error("No dataset uploaded. Please upload a dataset before proceeding.")
        # else:
        #     if st.session_state.remaining_questions:
        #         selected_question = st.session_state.remaining_questions[0]
        #         agent = create_pandas_dataframe_agent(llm, st.session_state.df_cleaned, verbose=True, allow_dangerous_code=True)
        #         response = agent.run(selected_question)
        #         st.subheader("Analysis Result:")
        #         st.write(response)
                
        #         fig, ax = plt.subplots()
        #         if "income" in selected_question:
        #             st.session_state.df_cleaned['income'] = st.session_state.df_cleaned['income'].astype(str)
        #             df_grouped = st.session_state.df_cleaned.groupby(['income']).size()
        #             df_grouped.plot(kind='bar', ax=ax)
        #             ax.set_title("Income Distribution")
        #             ax.set_ylabel("Count")
        #             ax.set_xlabel("Income Category")
                
        #         elif "age" in selected_question:
        #             df_grouped = st.session_state.df_cleaned.groupby(['age', 'income']).size().unstack()
        #             df_grouped.plot(kind='line', ax=ax)
        #             ax.set_title("Income Distribution by Age")
        #             ax.set_ylabel("Count")
        #             ax.set_xlabel("Age")
                
        #         elif "gender" in selected_question:
        #             df_grouped = st.session_state.df_cleaned.groupby(['sex', 'income']).size().unstack()
        #             df_grouped.plot(kind='bar', stacked=True, ax=ax)
        #             ax.set_title("Income Distribution by Gender")
        #             ax.set_ylabel("Count")
        #             ax.set_xlabel("Gender")
                
        #         elif "ethnic" in selected_question:
        #             df_grouped = st.session_state.df_cleaned.groupby(['race', 'income']).size().unstack()
        #             df_grouped.plot(kind='bar', stacked=True, ax=ax)
        #             ax.set_title("Income Distribution by Ethnic Group")
        #             ax.set_ylabel("Count")
        #             ax.set_xlabel("Ethnic Group")
                
        #         elif "country of origin" in selected_question:
        #             df_grouped = st.session_state.df_cleaned.groupby(['native-country', 'income']).size().unstack()
        #             df_grouped.plot(kind='barh', stacked=True, ax=ax)
        #             ax.set_title("Income Distribution by Country of Origin")
        #             ax.set_ylabel("Country of Origin")
        #             ax.set_xlabel("Count")
                
        #         st.pyplot(fig)
        #         st.session_state.completed_analyses.append((selected_question, response, fig))
                
        #         if st.button("Continue with another question"):
        #             st.session_state.remaining_questions.pop(0)
        #             st.rerun()
                
        #         if not st.session_state.remaining_questions:
        #             st.success("All questions have been analyzed!")
                    
        #             if st.button("Save results as PDF"):
        #                 pdf = FPDF()
        #                 pdf.set_auto_page_break(auto=True, margin=15)
        #                 pdf.add_page()
        #                 pdf.set_font("Arial", size=12)
        #                 pdf.cell(200, 10, "Analysis Report", ln=True, align='C')
                        
        #                 for idx, (question, analysis, fig) in enumerate(st.session_state.completed_analyses):
        #                     pdf.add_page()
        #                     pdf.cell(0, 10, f"{idx+1}. {question}", ln=True)
        #                     pdf.multi_cell(0, 10, analysis)
                            
        #                     image_path = f"temp_chart_{idx}.png"
        #                     fig.savefig(image_path)
        #                     pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=180)
        #                     os.remove(image_path)
                        
        #                 pdf_output_path = "analysis_report.pdf"
        #                 pdf.output(pdf_output_path)
        #                 with open(pdf_output_path, "rb") as pdf_file:
        #                     st.download_button(label="Download PDF Report", data=pdf_file, file_name="analysis_report.pdf", mime="application/pdf")
        if st.session_state.cleaned:
            st.divider()
            st.header("Data Science Problem")
            st.write("Now that we have digged deeper into our data, let's define a data science problem.")

            with st.sidebar:
                with st.expander("The importance of framing a data science problem approriately"):
                    st.caption(data_science_framing())


            prompt = st.text_input("Add your science problem here")


            if prompt:
                my_chain_output = chains_output(prompt)
                my_data_problem = my_chain_output[0].content if hasattr(my_chain_output[0], 'content') else str(my_chain_output[0])
                my_model_selection = my_chain_output[1].content if hasattr(my_chain_output[1], 'content') else str(my_chain_output[1])

            
                st.subheader("Data Science Problem:")
                st.write(my_data_problem)
                st.subheader("Suggested Machine Learning Models:")
                st.write(my_model_selection)
            
                with st.sidebar:
                    with st.expander("Which algorithm?"):
                        st.caption(algortihm_selection())

            
                formatted_list = list_to_selectbox(my_model_selection)
                selected_algorithm = st.selectbox("Select machine learning algorithm", formatted_list)

                if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                    st.subheader(f"Python Solution using {selected_algorithm}")
                    
                    with st.spinner('Generating solution code... This may take a moment.'):
                        try:
                            solution = python_solution(my_data_problem, selected_algorithm, st.session_state.df_cleaned)
                            
                            # Display the code with syntax highlighting
                            st.code(solution, language='python')
                            
                            # Add a download button
                            st.download_button(
                                label="Download Python Code",
                                data=solution,
                                file_name=f"{selected_algorithm.replace(' ', '_').lower()}_solution.py",
                                mime="text/plain"
                            )
                            
                            # Add execute button with a debug indicator
                            if st.button("Execute Code and Show Results", key="execute_code_btn"):
                                
                                # Execute the code with our simplified function
                                with st.spinner('Executing code... This may take a moment.'):
                                    output, error, figures, metrics = execute_python_code(solution, st.session_state.df_cleaned)

                                
                                # Display the results
                                display_execution_results(output, error, figures, metrics)
                            
                            # Add an explanation section
                            with st.expander("Code Explanation"):
                                explanation_prompt = f"Explain the following {selected_algorithm} code in simple terms:\n\n{solution[:1000]}..."
                                explanation = llm.invoke(explanation_prompt).content
                                st.write(explanation)
                                
                        except Exception as e:
                            st.error(f"Error generating solution: {str(e)}")
                            st.info("Try selecting a different algorithm or refreshing the page.")