import streamlit as st
import pandas as pd
from config import Config, CLEANING_ACTIONS
from domain_detector import detect_domain, get_domain_specific_guidelines
from plan_generator import generate_initial_plan, finalize_plan, validate_plan_execution, get_plan_summary
from data_cleaner import execute_cleaning_plan
from utils import validate_csv, generate_download_link, format_actions_display, calculate_metrics, get_data_preview_stats, display_metrics_comparison

Config.setup_page()

st.title("AI Data Cleaning Tool")
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue")
    st.stop()

try:
    chat_model = Config.get_groq_model(groq_api_key)
except:
    st.error("Invalid Groq API key")
    st.stop()

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    is_valid, validation_msg = validate_csv(uploaded_file)
    if not is_valid:
        st.error(validation_msg)
        st.stop()
    
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        with st.spinner("Analyzing dataset domain..."):
            domain_info = detect_domain(df, chat_model)
        
        st.subheader("Data Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Domain", domain_info['domain'])
        with col2:
            st.metric("Confidence", domain_info['confidence'])
        with col3:
            st.metric("Rows x Columns", f"{df.shape[0]} x {df.shape[1]}")
        
        preview_stats = get_data_preview_stats(df)
        with st.expander("Detailed Dataset Stats"):
            st.json(preview_stats)
        
        with st.spinner("Generating cleaning plan..."):
            cleaning_plan, initial_eda = generate_initial_plan(df, domain_info, chat_model)
        
        st.subheader("AI Cleaning Plan")
        
        if cleaning_plan.get('is_clean', False):
            st.warning("AI suggests your data is already clean. Proceed with caution.")
        
        st.write(f"Assessment: {cleaning_plan.get('message', 'No assessment available')}")
        st.write(f"Cleanliness Score: {cleaning_plan.get('cleanliness_score', 'N/A')}/100")
        
        st.subheader("Recommended Actions")
        actions = cleaning_plan.get('recommended_actions', [])
        modified_actions = []
        
        for i, action in enumerate(actions):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"{action['action'].replace('_', ' ').title()}")
                st.write(f"{action['description']}")
                st.write(f"Columns: {action['columns']}")
            with col2:
                st.write(f"Priority: {action['priority']}")
                st.write(f"Impact: {action.get('expected_impact', 'N/A')}")
            with col3:
                include = st.checkbox("Include", value=True, key=f"include_{i}")
            
            if include:
                modified_actions.append(action)
        
        st.subheader("Custom Actions")
        custom_actions = st.text_area("Add additional cleaning actions (one per line):")
        
        if st.button("Finalize Plan") or 'final_plan' in st.session_state:
            if custom_actions:
                for custom_action in custom_actions.split('\n'):
                    if custom_action.strip():
                        modified_actions.append({
                            "action": "custom_action",
                            "description": custom_action.strip(),
                            "columns": "all",
                            "priority": "medium"
                        })
            
            user_modifications = {
                "included_actions": [act['action'] for act in modified_actions],
                "custom_actions": custom_actions
            }
            
            final_plan = finalize_plan(cleaning_plan, user_modifications, initial_eda, chat_model)
            st.session_state.final_plan = final_plan
            st.session_state.original_df = df.copy()
            
            st.success("Plan finalized. Ready to execute cleaning.")
            
            plan_summary = get_plan_summary(final_plan)
            st.write(f"Total Actions: {plan_summary['total_actions']}")
            st.write(f"Estimated Time: {plan_summary['estimated_time']}")
            st.write(f"Risk Level: {plan_summary['risk_level']}")
        
        if 'final_plan' in st.session_state and st.button("Execute Cleaning Plan"):
            is_valid, validation_msg = validate_plan_execution(st.session_state.final_plan, st.session_state.original_df)
            if not is_valid:
                st.error(validation_msg)
            else:
                with st.spinner("Executing cleaning plan..."):
                    cleaned_df, execution_log = execute_cleaning_plan(
                    st.session_state.original_df, 
                    st.session_state.final_plan,
                    domain_info )
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.execution_log = execution_log
                
                metrics = calculate_metrics(st.session_state.original_df, cleaned_df)
                display_metrics_comparison(metrics)
                
                st.subheader("Execution Log")
                log_df = pd.DataFrame(execution_log)
                st.dataframe(log_df)
                
                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_df.head(100))
                
                csv_data = generate_download_link(cleaned_df)
                st.download_button(
                    label="Download Cleaned CSV",
                    data=csv_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("Upload a CSV file to analyze and clean your data using AI-powered domain detection and cleaning recommendations.")