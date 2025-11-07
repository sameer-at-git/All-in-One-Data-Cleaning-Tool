# Data Cleaning App Development Plan

## Project Structure

data_cleaning_app/
├── app.py # Main Streamlit application
├── config.py # Configuration and constants
├── domain_detector.py # Domain detection logic
├── plan_generator.py # LLM agents for generating cleaning plans
├── data_cleaner.py # Data cleaning execution logic
├── utils.py # Utility functions
└── requirements.txt # Dependencies

## Core Modules

### 1. config.py

- API configuration management
- Constants and default values
- Streamlit page configuration

### 2. domain_detector.py

**Functions:**

- detect_domain(df, chat_model) - Analyzes dataset to determine domain
- extract_dataset_info(df) - Extracts column names, data types, sample data
- Returns: domain, confidence level, reasoning

### 3. plan_generator.py

**LLM Agent 1 - Analysis & Planning:**

- generate_initial_plan(df, domain_info, chat_model) - Creates initial cleaning plan
- Performs initial EDA analysis
- Assesses data cleanliness
- Generates domain-specific action recommendations

**LLM Agent 2 - Plan Finalization:**

- finalize_plan(original_plan, user_modifications, initial_eda, chat_model)
- Incorporates user feedback
- Creates executable action sequence

### 4. data_cleaner.py

**Cleaning Operations:**

- execute_cleaning_plan(df, final_plan) - Main execution function
- handle_missing_values(df, columns) - Missing data treatment
- remove_duplicates(df) - Duplicate removal
- fix_data_types(df, columns) - Type conversion
- standardize_format(df, columns) - Text standardization
- remove_outliers(df, columns) - Outlier detection and removal
- encode_categorical(df, columns) - Categorical encoding

### 5. utils.py

**Helper Functions:**

- validate_csv(file) - CSV validation
- generate_download_link(df) - Download preparation
- format_actions_display(actions) - UI formatting
- calculate_metrics(before_df, after_df) - Progress metrics

### 6. app.py

**Streamlit UI Components:**

- API key input section
- File upload handler
- Domain detection display
- Interactive plan modification interface
- Real-time EDA visualization
- Cleaning execution progress
- Results display and download section

## Workflow Sequence

1. **File Upload & Validation**

   - User uploads CSV
   - Validate file format and size
   - Load into pandas DataFrame

2. **Domain Detection**

   - Extract column info and sample data
   - Use LLM to detect domain
   - Display domain confidence

3. **Initial Analysis & Planning**

   - Perform basic EDA (shape, nulls, dtypes)
   - LLM Agent 1 generates cleaning plan
   - Assess data cleanliness level

4. **User Interaction**

   - Display proposed actions
   - Allow add/remove/modify actions
   - LLM Agent 2 finalizes plan

5. **Execution Phase**

   - Execute cleaning actions sequentially
   - Track progress and success
   - Handle errors gracefully

6. **Results Delivery**

   - Display before/after metrics
   - Show cleaned data preview
   - Provide download option

## Key Features

- Modular and scalable architecture
- Support for large datasets (chunk processing)
- Real-time progress tracking
- Interactive plan customization
- Comprehensive error handling
- Clean separation of concerns

## Dependencies

- streamlit
- pandas
- numpy
- langchain-groq
- python-dotenv (optional for env vars)

This modular approach ensures maintainability, testability, and easy extension of functionality.
