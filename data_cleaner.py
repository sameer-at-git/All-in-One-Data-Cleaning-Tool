import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import json

def execute_cleaning_plan(df, final_plan, domain_info=None):
    execution_log = []
    cleaned_df = df.copy()
    
    for action in final_plan.get("finalized_actions", []):
        action_name = action["action"]
        columns = action["columns"]
        
        try:
            if action_name == "handle_missing_values":
                cleaned_df = handle_missing_values(cleaned_df, columns, domain_info)
            elif action_name == "remove_duplicates":
                cleaned_df = remove_duplicates(cleaned_df)
            elif action_name == "fix_data_types":
                cleaned_df = fix_data_types(cleaned_df, columns)
            elif action_name == "standardize_format":
                cleaned_df = standardize_format(cleaned_df, columns)
            elif action_name == "remove_outliers":
                cleaned_df = remove_outliers(cleaned_df, columns)
            elif action_name == "encode_categorical":
                cleaned_df = encode_categorical(cleaned_df, columns)
            elif action_name == "normalize_numeric":
                cleaned_df = normalize_numeric(cleaned_df, columns)
            elif action_name == "standardize_date_format":
                cleaned_df = standardize_date_format(cleaned_df, columns)
            elif action_name == "extract_features":
                cleaned_df = extract_features(cleaned_df, columns)
            elif action_name == "remove_columns":
                cleaned_df = remove_columns(cleaned_df, columns)
            elif action_name == "rename_columns":
                cleaned_df = rename_columns(cleaned_df, columns)
            elif action_name == "handle_inconsistent_casing":
                cleaned_df = handle_inconsistent_casing(cleaned_df, columns)
            elif action_name == "remove_special_characters":
                cleaned_df = remove_special_characters(cleaned_df, columns)
            elif action_name == "validate_email_format":
                cleaned_df = validate_email_format(cleaned_df, columns)
            elif action_name == "validate_phone_format":
                cleaned_df = validate_phone_format(cleaned_df, columns)
            elif action_name == "handle_currency_format":
                cleaned_df = handle_currency_format(cleaned_df, columns)
            elif action_name == "convert_units":
                cleaned_df = convert_units(cleaned_df, columns)
            elif action_name == "handle_skewness":
                cleaned_df = handle_skewness(cleaned_df, columns)
            elif action_name == "bin_numeric_variables":
                cleaned_df = bin_numeric_variables(cleaned_df, columns)
            elif action_name == "handle_text_encoding":
                cleaned_df = handle_text_encoding(cleaned_df, columns)
            elif action_name == "remove_whitespace":
                cleaned_df = remove_whitespace(cleaned_df, columns)
            elif action_name == "validate_postal_codes":
                cleaned_df = validate_postal_codes(cleaned_df, columns)
            elif action_name == "handle_country_names":
                cleaned_df = handle_country_names(cleaned_df, columns)
            elif action_name == "extract_datetime_components":
                cleaned_df = extract_datetime_components(cleaned_df, columns)
            elif action_name == "handle_abbreviations":
                cleaned_df = handle_abbreviations(cleaned_df, columns)
            elif action_name == "detect_anomalies":
                cleaned_df = detect_anomalies(cleaned_df, columns)
            elif action_name == "handle_zero_values":
                cleaned_df = handle_zero_values(cleaned_df, columns)
            elif action_name == "standardize_boolean":
                cleaned_df = standardize_boolean(cleaned_df, columns)
            elif action_name == "handle_infinite_values":
                cleaned_df = handle_infinite_values(cleaned_df, columns)
            elif action_name == "validate_ranges":
                cleaned_df = validate_ranges(cleaned_df, columns)
            elif action_name == "handle_negative_values":
                cleaned_df = handle_negative_values(cleaned_df, columns)
            elif action_name == "create_derived_features":
                cleaned_df = create_derived_features(cleaned_df, columns)
            elif action_name == "handle_multiple_categories":
                cleaned_df = handle_multiple_categories(cleaned_df, columns)
            elif action_name == "standardize_address_format":
                cleaned_df = standardize_address_format(cleaned_df, columns)
            elif action_name == "validate_urls":
                cleaned_df = validate_urls(cleaned_df, columns)
            elif action_name == "handle_percentages":
                cleaned_df = handle_percentages(cleaned_df, columns)
            elif action_name == "remove_irrelevant_columns":
                cleaned_df = remove_irrelevant_columns(cleaned_df, columns)
            elif action_name == "handle_correlated_features":
                cleaned_df = handle_correlated_features(cleaned_df, columns)
            elif action_name == "standardize_names":
                cleaned_df = standardize_names(cleaned_df, columns)
            elif action_name == "handle_ordinal_categories":
                cleaned_df = handle_ordinal_categories(cleaned_df, columns)
            
            execution_log.append({
                "action": action_name,
                "success": True,
                "rows_after": cleaned_df.shape[0],
                "columns_after": cleaned_df.shape[1]
            })
            
        except Exception as e:
            execution_log.append({
                "action": action_name,
                "success": False,
                "error": str(e),
                "rows_after": cleaned_df.shape[0],
                "columns_after": cleaned_df.shape[1]
            })
    
    return cleaned_df, execution_log

def handle_missing_values(df, columns, domain_info=None):
    if columns == "all":
        columns = df.columns
    
    domain = domain_info.get('domain', 'general') if domain_info else 'general'
    
    for col in columns:
        if col in df.columns:
            col_dtype = df[col].dtype
            col_name_lower = col.lower()
            
            if col_dtype in ['object', 'string']:
                if any(keyword in col_name_lower for keyword in ['date', 'time']):
                    df[col].fillna('Unknown Date', inplace=True)
                elif any(keyword in col_name_lower for keyword in ['name', 'title', 'description']):
                    df[col].fillna('Unknown', inplace=True)
                elif any(keyword in col_name_lower for keyword in ['email', 'phone', 'id']):
                    df[col].fillna('Not Provided', inplace=True)
                elif domain in ['finance', 'sales', 'ecommerce'] and any(keyword in col_name_lower for keyword in ['category', 'type', 'status']):
                    df[col].fillna('Other', inplace=True)
                else:
                    df[col].fillna('Missing', inplace=True)
                    
            elif col_dtype in ['datetime64[ns]']:
                df[col].fillna(pd.NaT, inplace=True)
                
            elif col_dtype in ['int64', 'float64']:
                if any(keyword in col_name_lower for keyword in ['price', 'amount', 'cost', 'revenue', 'salary', 'income']):
                    df[col].fillna(0, inplace=True)
                elif any(keyword in col_name_lower for keyword in ['age', 'year', 'count', 'quantity']):
                    df[col].fillna(df[col].median(), inplace=True)
                elif any(keyword in col_name_lower for keyword in ['rating', 'score', 'percentage']):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def fix_data_types(df, columns):
    if columns == "all":
        columns = df.columns
    
    for col in columns:
        if col in df.columns:
            col_name_lower = col.lower()
            
            if df[col].dtype == 'object':
                if any(keyword in col_name_lower for keyword in ['date', 'time', 'year']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
                elif any(keyword in col_name_lower for keyword in ['price', 'amount', 'cost', 'revenue', 'salary']):
                    df[col] = df[col].replace('[\$,]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif any(keyword in col_name_lower for keyword in ['percentage', 'rate']):
                    df[col] = df[col].replace('%', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                elif df[col].str.contains('^\d+$').all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def standardize_format(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            col_name_lower = col.lower()
            
            if any(keyword in col_name_lower for keyword in ['email']):
                df[col] = df[col].astype(str).str.lower().strip()
            elif any(keyword in col_name_lower for keyword in ['name', 'title']):
                df[col] = df[col].astype(str).str.title().strip()
            elif any(keyword in col_name_lower for keyword in ['address', 'location']):
                df[col] = df[col].astype(str).str.upper().strip()
            else:
                df[col] = df[col].astype(str).str.strip()
    
    return df

def remove_outliers(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['number']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def encode_categorical(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            if df[col].nunique() <= 10:
                df = pd.get_dummies(df, columns=[col], prefix=[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    return df

def normalize_numeric(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['number']).columns
    
    scaler = StandardScaler()
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            df[col] = scaler.fit_transform(df[[col]])
    return df

def standardize_date_format(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['object']).columns
    
    date_patterns = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d'
    ]
    
    for col in columns:
        if col in df.columns:
            for pattern in date_patterns:
                try:
                    df[col] = pd.to_datetime(df[col], format=pattern, errors='ignore')
                except:
                    continue
    return df

def extract_features(df, columns):
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[f'{col}_length'] = df[col].str.len()
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
    return df

def remove_columns(df, columns):
    return df.drop(columns=columns, errors='ignore')

def rename_columns(df, columns):
    rename_dict = {}
    for col in columns:
        if col in df.columns:
            new_name = col.lower().replace(' ', '_').replace('-', '_')
            rename_dict[col] = new_name
    return df.rename(columns=rename_dict)

def handle_inconsistent_casing(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.title()
    return df

def remove_special_characters(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
    return df

def validate_email_format(df, columns):
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[f'{col}_valid'] = df[col].astype(str).str.match(email_pattern)
    return df

def validate_phone_format(df, columns):
    phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            clean_phone = df[col].astype(str).str.replace(r'[\s\(\)\-]', '', regex=True)
            df[f'{col}_valid'] = clean_phone.str.match(phone_pattern)
    return df

def handle_currency_format(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def convert_units(df, columns):
    conversion_factors = {
        'kg_to_lb': 2.20462,
        'lb_to_kg': 0.453592,
        'km_to_mile': 0.621371,
        'mile_to_km': 1.60934
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            df[f'{col}_converted'] = df[col] * conversion_factors.get('kg_to_lb', 1)
    return df

def handle_skewness(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['number']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                df[col] = np.log1p(df[col])
    return df

def bin_numeric_variables(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            df[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    return df

def handle_text_encoding(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
    return df

def remove_whitespace(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

def validate_postal_codes(df, columns):
    postal_pattern = r'^[A-Z0-9\-\s]{3,10}$'
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[f'{col}_valid'] = df[col].astype(str).str.upper().str.match(postal_pattern)
    return df

def handle_country_names(df, columns):
    country_mapping = {
        'usa': 'United States', 'us': 'United States', 'u.s.a': 'United States',
        'uk': 'United Kingdom', 'u.k': 'United Kingdom', 'england': 'United Kingdom',
        'uae': 'United Arab Emirates'
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.lower().map(country_mapping).fillna(df[col])
    return df

def extract_datetime_components(df, columns):
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    return df

def handle_abbreviations(df, columns):
    abbreviation_mapping = {
        'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
        'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place'
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            for abbr, full in abbreviation_mapping.items():
                df[col] = df[col].astype(str).str.replace(rf'\b{abbr}\b', full, regex=True)
    return df

def detect_anomalies(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['number']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[f'{col}_anomaly'] = z_scores > 3
    return df

def handle_zero_values(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            zero_mask = df[col] == 0
            if zero_mask.any():
                df.loc[zero_mask, col] = df[col].median()
    return df

def standardize_boolean(df, columns):
    bool_mapping = {
        'yes': True, 'no': False, 'true': True, 'false': False,
        '1': True, '0': False, 'y': True, 'n': False
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.lower().map(bool_mapping).fillna(df[col])
    return df

def handle_infinite_values(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['number']).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col].fillna(df[col].median(), inplace=True)
    return df

def validate_ranges(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            df[f'{col}_in_range'] = (df[col] >= df[col].quantile(0.01)) & (df[col] <= df[col].quantile(0.99))
    return df

def handle_negative_values(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            negative_mask = df[col] < 0
            if negative_mask.any() and col.lower() not in ['profit', 'growth', 'change']:
                df.loc[negative_mask, col] = abs(df.loc[negative_mask, col])
    return df

def create_derived_features(df, columns):
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 2:
        df['feature_ratio'] = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1e-8)
        df['feature_sum'] = df[numeric_cols[0]] + df[numeric_cols[1]]
    return df

def handle_multiple_categories(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            value_counts = df[col].value_counts()
            top_categories = value_counts.head(10).index
            df[col] = df[col].where(df[col].isin(top_categories), 'Other')
    return df

def standardize_address_format(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.upper().str.strip()
    return df

def validate_urls(df, columns):
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[f'{col}_valid'] = df[col].astype(str).str.match(url_pattern, case=False)
    return df

def handle_percentages(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100
    return df

def remove_irrelevant_columns(df, columns):
    irrelevant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            irrelevant_cols.append(col)
    return df.drop(columns=irrelevant_cols)

def handle_correlated_features(df, columns):
    if columns == "all":
        columns = df.select_dtypes(include=['number']).columns
    
    correlation_matrix = df[columns].corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    return df.drop(columns=to_drop)

def standardize_names(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.title().str.strip()
    return df

def handle_ordinal_categories(df, columns):
    ordinal_mappings = {
        'size': ['small', 'medium', 'large', 'x-large'],
        'quality': ['poor', 'fair', 'good', 'excellent'],
        'priority': ['low', 'medium', 'high', 'critical']
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            for category, levels in ordinal_mappings.items():
                if category in col.lower():
                    df[col] = pd.Categorical(df[col], categories=levels, ordered=True)
                    break
    return df