import streamlit as st

class Config:
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    MAX_FILE_SIZE = 200 * 1024 * 1024
    SAMPLE_ROWS = 3
    SUPPORTED_DOMAINS = ["sales", "users", "weather", "healthcare", "finance", "ecommerce", "education", "general"]
    
    @staticmethod
    def get_groq_model(api_key):
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=api_key, model_name=Config.DEFAULT_MODEL)
    
    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title="AI Data Cleaner",
            page_icon=":robot:",
            layout="wide",
            initial_sidebar_state="expanded"
        )

CLEANING_ACTIONS = {
    "handle_missing_values": "Handle missing data",
    "remove_duplicates": "Remove duplicate rows",
    "fix_data_types": "Fix data type inconsistencies",
    "standardize_format": "Standardize text formats",
    "remove_outliers": "Remove statistical outliers",
    "encode_categorical": "Encode categorical variables",
    "normalize_numeric": "Normalize numerical columns",
    "standardize_date_format": "Standardize date formats",
    "extract_features": "Extract features from complex columns",
    "remove_columns": "Remove unnecessary columns",
    "rename_columns": "Rename columns to standard format",
    "handle_inconsistent_casing": "Fix inconsistent text casing",
    "remove_special_characters": "Remove special characters from text",
    "validate_email_format": "Validate and fix email formats",
    "validate_phone_format": "Validate and fix phone number formats",
    "handle_currency_format": "Standardize currency formats",
    "convert_units": "Convert measurement units to standard",
    "handle_skewness": "Handle skewed numerical distributions",
    "bin_numeric_variables": "Bin numerical variables into categories",
    "handle_text_encoding": "Fix text encoding issues",
    "remove_whitespace": "Remove extra whitespace from text",
    "validate_postal_codes": "Validate and format postal codes",
    "handle_country_names": "Standardize country names and codes",
    "extract_datetime_components": "Extract year, month, day from dates",
    "handle_abbreviations": "Standardize common abbreviations",
    "detect_anomalies": "Detect and handle data anomalies",
    "handle_zero_values": "Handle zero values appropriately",
    "standardize_boolean": "Standardize boolean representations",
    "handle_infinite_values": "Handle infinite numerical values",
    "validate_ranges": "Validate data within expected ranges",
    "handle_negative_values": "Handle unexpected negative values",
    "create_derived_features": "Create new features from existing ones",
    "handle_multiple_categories": "Handle too many categorical values",
    "standardize_address_format": "Standardize address formats",
    "validate_urls": "Validate and format URLs",
    "handle_percentages": "Standardize percentage formats",
    "remove_irrelevant_columns": "Remove columns with single values",
    "handle_correlated_features": "Handle highly correlated features",
    "standardize_names": "Standardize person and company names",
    "handle_ordinal_categories": "Properly encode ordinal categories",
    "detect_collisions": "Detect and handle ID collisions",
    "validate_ages": "Validate and handle age values",
    "handle_timezones": "Standardize timezone information",
    "remove_html_tags": "Remove HTML tags from text",
    "handle_json_columns": "Parse and flatten JSON columns",
    "standardize_measurements": "Standardize measurement formats",
    "validate_lat_long": "Validate latitude and longitude values",
    "handle_array_columns": "Handle array-type data columns",
    "detect_data_drift": "Detect and handle data drift issues",
    "handle_imbalanced_data": "Address class imbalance issues",
    "standardize_product_codes": "Standardize product/SKU codes",
    "validate_credit_cards": "Validate and format credit card numbers",
    "handle_multi_value_columns": "Split multi-value columns",
    "standardize_department_names": "Standardize department/organization names",
    "validate_social_security": "Validate and format SSN/ID numbers",
    "handle_nested_structures": "Handle nested data structures",
    "standardize_job_titles": "Standardize job title formats",
    "validate_ip_addresses": "Validate and format IP addresses",
    "handle_legacy_data": "Handle legacy data format conversions",
    "standardize_education_levels": "Standardize education level formats",
    "validate_business_codes": "Validate business classification codes",
    "handle_geospatial_data": "Process geospatial data formats",
    "standardize_marital_status": "Standardize marital status formats",
    "validate_vat_numbers": "Validate VAT/tax identification numbers",
    "handle_historical_data": "Handle historical data inconsistencies",
    "standardize_gender_codes": "Standardize gender representation",
    "validate_passport_numbers": "Validate passport number formats",
    "handle_multilingual_data": "Handle multilingual text data",
    "standardize_industry_codes": "Standardize industry classification codes"
}
