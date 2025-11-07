import pandas as pd
import streamlit as st
import io
import chardet

def validate_csv(file):
    if file is None:
        return False, "No file uploaded"
    
    if not file.name.lower().endswith('.csv'):
        return False, "File must be a CSV"
    
    try:
        file_size = len(file.getvalue())
        if file_size > 200 * 1024 * 1024:
            return False, "File size exceeds 200MB limit"
        
        raw_data = file.getvalue()
        encoding = chardet.detect(raw_data)['encoding']
        
        file.seek(0)
        sample_data = file.read(1024)
        file.seek(0)
        
        if b'\0' in sample_data:
            return False, "File appears to be binary or corrupted"
        
        return True, "CSV validation successful"
    
    except Exception as e:
        return False, f"Error validating CSV: {str(e)}"

def generate_download_link(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    
    return csv_str

def format_actions_display(actions):
    formatted_actions = []
    
    for action in actions:
        formatted_action = {
            "action": action.get("action", "").replace("_", " ").title(),
            "description": action.get("description", ""),
            "columns": ", ".join(action["columns"]) if isinstance(action.get("columns"), list) else action.get("columns", "all"),
            "priority": action.get("priority", "").title(),
            "reasoning": action.get("reasoning", ""),
            "expected_impact": action.get("expected_impact", "")
        }
        formatted_actions.append(formatted_action)
    
    return formatted_actions

def calculate_metrics(before_df, after_df):
    metrics = {
        "original_rows": before_df.shape[0],
        "cleaned_rows": after_df.shape[0],
        "original_columns": before_df.shape[1],
        "cleaned_columns": after_df.shape[1],
        "rows_removed": before_df.shape[0] - after_df.shape[0],
        "columns_removed": before_df.shape[1] - after_df.shape[1],
        "original_memory_usage": before_df.memory_usage(deep=True).sum(),
        "cleaned_memory_usage": after_df.memory_usage(deep=True).sum(),
        "memory_reduction": before_df.memory_usage(deep=True).sum() - after_df.memory_usage(deep=True).sum(),
        "original_null_count": before_df.isnull().sum().sum(),
        "cleaned_null_count": after_df.isnull().sum().sum(),
        "null_reduction": before_df.isnull().sum().sum() - after_df.isnull().sum().sum(),
        "original_duplicates": before_df.duplicated().sum(),
        "cleaned_duplicates": after_df.duplicated().sum(),
        "duplicates_removed": before_df.duplicated().sum() - after_df.duplicated().sum()
    }
    
    metrics["row_reduction_percent"] = (metrics["rows_removed"] / metrics["original_rows"] * 100) if metrics["original_rows"] > 0 else 0
    metrics["memory_reduction_percent"] = (metrics["memory_reduction"] / metrics["original_memory_usage"] * 100) if metrics["original_memory_usage"] > 0 else 0
    metrics["null_reduction_percent"] = (metrics["null_reduction"] / metrics["original_null_count"] * 100) if metrics["original_null_count"] > 0 else 0
    metrics["duplicate_reduction_percent"] = (metrics["duplicates_removed"] / metrics["original_duplicates"] * 100) if metrics["original_duplicates"] > 0 else 0
    
    return metrics

def get_data_preview_stats(df):
    stats = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "data_types": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "null_percentage": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
        "duplicate_percentage": float((df.duplicated().sum() / df.shape[0]) * 100)
    }
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats["numeric_columns"] = len(numeric_cols)
        stats["numeric_stats"] = {col: {k: float(v) for k, v in df[col].describe().to_dict().items()} for col in numeric_cols}
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        stats["categorical_columns"] = len(categorical_cols)
        stats["unique_values_per_cat"] = {col: int(df[col].nunique()) for col in categorical_cols}
    
    return stats

def format_file_size(bytes_size):
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"

def create_progress_tracker(total_steps):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(step, description):
        progress = (step + 1) / total_steps
        progress_bar.progress(progress)
        status_text.text(f"Step {step + 1}/{total_steps}: {description}")
    
    return update_progress

def cleanup_resources():
    if 'uploaded_file' in st.session_state:
        del st.session_state.uploaded_file
    if 'original_df' in st.session_state:
        del st.session_state.original_df
    if 'cleaned_df' in st.session_state:
        del st.session_state.cleaned_df
    if 'execution_log' in st.session_state:
        del st.session_state.execution_log

def display_metrics_comparison(metrics):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", metrics["cleaned_rows"], f"{metrics['rows_removed']} removed")
    
    with col2:
        st.metric("Columns", metrics["cleaned_columns"], f"{metrics['columns_removed']} removed")
    
    with col3:
        st.metric("Null Values", metrics["cleaned_null_count"], f"{metrics['null_reduction']} removed")
    
    with col4:
        st.metric("Memory Usage", 
                 format_file_size(metrics["cleaned_memory_usage"]), 
                 f"-{metrics['memory_reduction_percent']:.1f}%")

def validate_dataframe_integrity(df):
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
    
    if df.isnull().all().any():
        issues.append("Some columns contain only null values")
    
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"Column '{col}' has only one unique value")
    
    if df.duplicated().sum() > df.shape[0] * 0.5:
        issues.append("More than 50% of rows are duplicates")
    
    return len(issues) == 0, issues