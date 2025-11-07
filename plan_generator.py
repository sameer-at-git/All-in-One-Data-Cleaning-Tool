import pandas as pd
import json
from langchain_core.messages import HumanMessage, SystemMessage
from config import CLEANING_ACTIONS
from domain_detector import get_domain_specific_guidelines

def generate_initial_plan(df, domain_info, chat_model):
    initial_eda = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "date_columns": df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    total_nulls = sum(initial_eda['null_counts'].values())
    total_duplicates = initial_eda['duplicate_rows']
    
    if total_nulls == 0 and total_duplicates == 0:
        clean_plan = {
            "is_clean": True,
            "cleanliness_score": 95,
            "message": "Data appears to be already clean - no null values or duplicates detected",
            "domain_specific_notes": f"High-quality {domain_info['domain']} dataset",
            "critical_issues": [],
            "recommended_actions": [],
            "warnings": ["Data is already clean. Consider if additional processing is needed"],
            "estimated_time": "No processing needed"
        }
        return clean_plan, initial_eda
    
    domain_guidelines = get_domain_specific_guidelines(domain_info['domain'])
    
    system_prompt = f"""You are a data cleaning expert specializing in {domain_info['domain']} data. Analyze the dataset and create a comprehensive cleaning plan.

Dataset Analysis:
- Shape: {initial_eda['shape']}
- Columns: {initial_eda['columns']}
- Null values: {initial_eda['null_counts']}
- Data types: {initial_eda['dtypes']}
- Duplicate rows: {initial_eda['duplicate_rows']}
- Numeric columns: {initial_eda['numeric_columns']}
- Categorical columns: {initial_eda['categorical_columns']}
- Date columns: {initial_eda['date_columns']}

Domain: {domain_info['domain']}
Domain Confidence: {domain_info['confidence']}
Domain Reasoning: {domain_info['reasoning']}

Available Cleaning Actions: {list(CLEANING_ACTIONS.keys())}
Domain-Specific Guidelines: {domain_guidelines}

Provide a JSON response with this exact structure:
{{
    "is_clean": true/false,
    "cleanliness_score": 0-100,
    "message": "Detailed assessment of data quality and cleanliness",
    "domain_specific_notes": "Specific considerations for {domain_info['domain']} domain",
    "critical_issues": ["list of critical data quality issues"],
    "recommended_actions": [
        {{
            "action": "action_name_from_available_actions",
            "description": "Detailed description of what this action will do",
            "columns": ["specific_columns_or_all"],
            "priority": "high/medium/low",
            "reasoning": "Why this action is needed for this dataset",
            "expected_impact": "What improvement this will bring"
        }}
    ],
    "warnings": ["Any warnings or limitations"],
    "estimated_time": "Estimated processing time"
}}

Score cleanliness based on: null values, duplicates, data types, and data quality."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a comprehensive cleaning plan for this {domain_info['domain']} dataset.")
    ]
    
    try:
        response = chat_model.invoke(messages)
        plan_data = json.loads(response.content)
        
        for action in plan_data.get("recommended_actions", []):
            if action["action"] in CLEANING_ACTIONS:
                action["description"] = CLEANING_ACTIONS[action["action"]]
        
        return plan_data, initial_eda
    except Exception as e:
        default_plan = {
            "is_clean": False,
            "cleanliness_score": 50,
            "message": "Basic cleaning recommended",
            "domain_specific_notes": f"General {domain_info['domain']} dataset",
            "critical_issues": ["Missing values present", "Potential duplicates"],
            "recommended_actions": [
                {
                    "action": "handle_missing_values",
                    "description": CLEANING_ACTIONS["handle_missing_values"],
                    "columns": "all",
                    "priority": "high",
                    "reasoning": "Dataset contains missing values",
                    "expected_impact": "Complete data records"
                },
                {
                    "action": "remove_duplicates",
                    "description": CLEANING_ACTIONS["remove_duplicates"],
                    "columns": "all",
                    "priority": "high",
                    "reasoning": "Duplicate rows detected",
                    "expected_impact": "Unique records only"
                }
            ],
            "warnings": ["Automatic fallback plan used"],
            "estimated_time": "Quick processing"
        }
        return default_plan, initial_eda

def finalize_plan(original_plan, user_modifications, initial_eda, chat_model):
    system_prompt = f"""Finalize the data cleaning plan based on user modifications and initial analysis.

Original Plan:
{json.dumps(original_plan, indent=2, default=str)}

Initial EDA:
{json.dumps(initial_eda, indent=2, default=str)}

User Modifications:
{json.dumps(user_modifications, indent=2, default=str)}

Create a final executable plan that incorporates all user changes while maintaining data integrity.
Focus on practical, implementable actions.

Provide the final plan in this exact JSON structure:
{{
    "finalized_actions": [
        {{
            "action": "action_name",
            "description": "Clear implementation description",
            "columns": ["specific_columns"],
            "priority": "high/medium/low",
            "parameters": {{}},
            "execution_order": 1,
            "validation_required": true/false
        }}
    ],
    "execution_sequence": [1, 2, 3...],
    "total_estimated_time": "time estimate",
    "risk_assessment": "Low/Medium/High",
    "success_criteria": ["list of success metrics"]
}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Create the final executable cleaning plan.")
    ]
    
    try:
        response = chat_model.invoke(messages)
        return json.loads(response.content)
    except Exception as e:
        finalized_actions = []
        execution_order = 1
        
        for action in original_plan.get("recommended_actions", []):
            if action["action"] in user_modifications.get("included_actions", []):
                finalized_actions.append({
                    "action": action["action"],
                    "description": action["description"],
                    "columns": action["columns"],
                    "priority": action["priority"],
                    "parameters": {},
                    "execution_order": execution_order,
                    "validation_required": True
                })
                execution_order += 1
        
        return {
            "finalized_actions": finalized_actions,
            "execution_sequence": list(range(1, execution_order)),
            "total_estimated_time": "Standard processing",
            "risk_assessment": "Low",
            "success_criteria": ["Missing values handled", "Duplicates removed", "Data types fixed"]
        }

def validate_plan_execution(final_plan, df):
    required_columns = set()
    for action in final_plan.get("finalized_actions", []):
        if action["columns"] != "all":
            for col in action["columns"]:
                if col not in df.columns:
                    return False, f"Column '{col}' not found in dataset"
                required_columns.add(col)
    return True, "Plan validation successful"

def get_plan_summary(final_plan):
    summary = {
        "total_actions": len(final_plan.get("finalized_actions", [])),
        "high_priority_actions": sum(1 for action in final_plan.get("finalized_actions", []) if action.get("priority") == "high"),
        "medium_priority_actions": sum(1 for action in final_plan.get("finalized_actions", []) if action.get("priority") == "medium"),
        "low_priority_actions": sum(1 for action in final_plan.get("finalized_actions", []) if action.get("priority") == "low"),
        "estimated_time": final_plan.get("total_estimated_time", "Unknown"),
        "risk_level": final_plan.get("risk_assessment", "Unknown")
    }
    return summary