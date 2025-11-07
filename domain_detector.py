import pandas as pd
import json
from langchain_core.messages import HumanMessage, SystemMessage
def extract_dataset_info(df, sample_rows=3):
    column_info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample_data": df.head(sample_rows).to_dict('records')
    }
    return column_info

def detect_domain(df, chat_model, sample_rows=3):
    column_info = extract_dataset_info(df, sample_rows)
    
    system_prompt = """Analyze the dataset structure and determine its domain based on:
    - Column names
    - Data types  
    - Sample data values
    
    Respond ONLY with a JSON format: {"domain": "detected_domain", "confidence": "high/medium/low", "reasoning": "brief explanation"}
    Available domains: sales, ecommerce, retail, finance, banking, insurance, healthcare, medical, pharmaceuticals, users, customers, marketing, advertising, weather, climate, environment, education, academic, research, real_estate, property, manufacturing, production, logistics, supply_chain, transportation, shipping, human_resources, hr, recruitment, telecommunications, telecom, energy, utilities, agriculture, farming, entertainment, media, sports, fitness, government, public_sector, tourism, hospitality, automotive, transportation, technology, IT, software, biotechnology, bioinformatics, social_media, networking, legal, law, construction, engineering, aerospace, aviation, maritime, naval, mining, resources, textiles, fashion, food_beverage, restaurant, general"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(column_info, indent=2))
    ]
    
    try:
        response = chat_model.invoke(messages)
        return json.loads(response.content)
    except Exception as e:
        return {"domain": "general", "confidence": "low", "reasoning": f"Error in detection: {str(e)}"}

def get_domain_specific_guidelines(domain):
    guidelines = {
        "sales": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "handle_currency_format", "validate_ranges", "standardize_names", "handle_skewness", "encode_categorical", "remove_outliers", "create_derived_features"],
        "ecommerce": ["handle_missing_values", "remove_duplicates", "standardize_product_codes", "handle_currency_format", "validate_urls", "standardize_format", "encode_categorical", "remove_outliers", "extract_features", "handle_multiple_categories"],
        "retail": ["handle_missing_values", "remove_duplicates", "standardize_product_codes", "handle_currency_format", "validate_ranges", "standardize_measurements", "encode_categorical", "remove_outliers", "bin_numeric_variables", "handle_imbalanced_data"],
        "finance": ["handle_missing_values", "remove_duplicates", "handle_currency_format", "validate_credit_cards", "validate_ranges", "remove_outliers", "handle_skewness", "normalize_numeric", "detect_anomalies", "validate_account_numbers"],
        "banking": ["handle_missing_values", "remove_duplicates", "handle_currency_format", "validate_credit_cards", "validate_ranges", "standardize_names", "encode_categorical", "remove_outliers", "detect_anomalies", "validate_social_security"],
        "insurance": ["handle_missing_values", "remove_duplicates", "handle_currency_format", "validate_ranges", "standardize_names", "encode_categorical", "remove_outliers", "handle_skewness", "validate_ages", "handle_imbalanced_data"],
        "healthcare": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_ages", "validate_ranges", "standardize_medical_codes", "encode_categorical", "remove_outliers", "handle_sensitive_data", "validate_medical_values"],
        "medical": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_ages", "validate_ranges", "standardize_medical_codes", "encode_categorical", "remove_outliers", "handle_lab_results", "validate_measurements"],
        "pharmaceuticals": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_ranges", "standardize_drug_codes", "encode_categorical", "remove_outliers", "handle_chemical_data", "validate_concentrations", "standardize_units"],
        "users": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_phone_format", "standardize_names", "validate_ages", "standardize_location", "encode_categorical", "remove_special_characters", "handle_inconsistent_casing"],
        "customers": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_phone_format", "standardize_names", "validate_ages", "standardize_location", "encode_categorical", "segment_customers", "create_derived_features"],
        "marketing": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_urls", "standardize_campaign_codes", "encode_categorical", "remove_outliers", "handle_skewness", "create_derived_features", "bin_numeric_variables"],
        "advertising": ["handle_missing_values", "remove_duplicates", "validate_urls", "standardize_campaign_codes", "handle_currency_format", "encode_categorical", "remove_outliers", "handle_skewness", "create_derived_features", "validate_impression_data"],
        "weather": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "create_derived_features", "handle_time_series"],
        "climate": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "create_derived_features", "handle_time_series"],
        "environment": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "encode_categorical", "validate_environmental_data"],
        "education": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_ages", "validate_grades", "standardize_course_codes", "encode_categorical", "remove_outliers", "handle_academic_data", "create_derived_features"],
        "academic": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_citation_data", "standardize_research_codes", "encode_categorical", "remove_outliers", "handle_research_metrics", "validate_publication_data", "create_derived_features"],
        "research": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_measurements", "standardize_experiment_codes", "encode_categorical", "remove_outliers", "handle_scientific_data", "validate_statistical_data", "create_derived_features"],
        "real_estate": ["handle_missing_values", "remove_duplicates", "handle_currency_format", "validate_ranges", "standardize_location", "encode_categorical", "remove_outliers", "handle_geospatial_data", "standardize_property_codes", "create_derived_features"],
        "property": ["handle_missing_values", "remove_duplicates", "handle_currency_format", "validate_ranges", "standardize_location", "encode_categorical", "remove_outliers", "handle_geospatial_data", "standardize_property_codes", "validate_property_data"],
        "manufacturing": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_ranges", "standardize_product_codes", "encode_categorical", "remove_outliers", "handle_quality_metrics", "standardize_measurements", "validate_production_data"],
        "production": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_ranges", "standardize_product_codes", "encode_categorical", "remove_outliers", "handle_quality_metrics", "standardize_measurements", "validate_manufacturing_data"],
        "logistics": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_tracking_numbers", "standardize_location", "encode_categorical", "remove_outliers", "handle_geospatial_data", "validate_shipment_data", "create_derived_features"],
        "supply_chain": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_inventory_codes", "standardize_location", "encode_categorical", "remove_outliers", "handle_geospatial_data", "validate_supply_data", "create_derived_features"],
        "transportation": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_vehicle_data", "standardize_location", "encode_categorical", "remove_outliers", "handle_geospatial_data", "validate_transport_data", "create_derived_features"],
        "shipping": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_tracking_numbers", "standardize_location", "encode_categorical", "remove_outliers", "handle_geospatial_data", "validate_shipment_data", "create_derived_features"],
        "human_resources": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_phone_format", "standardize_names", "validate_ages", "standardize_job_titles", "encode_categorical", "validate_salaries", "create_derived_features"],
        "hr": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_phone_format", "standardize_names", "validate_ages", "standardize_job_titles", "encode_categorical", "validate_salaries", "handle_employee_data"],
        "recruitment": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_phone_format", "standardize_names", "validate_ages", "standardize_job_titles", "encode_categorical", "validate_skills_data", "handle_applicant_data"],
        "telecommunications": ["handle_missing_values", "remove_duplicates", "validate_phone_format", "validate_ip_addresses", "standardize_plan_codes", "encode_categorical", "remove_outliers", "handle_usage_data", "validate_network_data", "create_derived_features"],
        "telecom": ["handle_missing_values", "remove_duplicates", "validate_phone_format", "validate_ip_addresses", "standardize_plan_codes", "encode_categorical", "remove_outliers", "handle_usage_data", "validate_network_data", "create_derived_features"],
        "energy": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "validate_energy_data", "create_derived_features"],
        "utilities": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "validate_utility_data", "create_derived_features"],
        "agriculture": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "validate_agricultural_data", "create_derived_features"],
        "farming": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "convert_units", "validate_ranges", "remove_outliers", "handle_geospatial_data", "standardize_measurements", "validate_crop_data", "create_derived_features"],
        "entertainment": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_media_formats", "standardize_genre_codes", "encode_categorical", "remove_outliers", "handle_ratings_data", "validate_entertainment_data", "create_derived_features"],
        "media": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_media_formats", "standardize_genre_codes", "encode_categorical", "remove_outliers", "handle_ratings_data", "validate_media_data", "create_derived_features"],
        "sports": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_sports_codes", "standardize_team_names", "encode_categorical", "remove_outliers", "handle_performance_data", "validate_sports_metrics", "create_derived_features"],
        "fitness": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_health_metrics", "standardize_exercise_codes", "encode_categorical", "remove_outliers", "handle_fitness_data", "validate_workout_data", "create_derived_features"],
        "government": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_government_codes", "standardize_department_names", "encode_categorical", "remove_outliers", "handle_public_data", "validate_government_data", "create_derived_features"],
        "public_sector": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_public_codes", "standardize_department_names", "encode_categorical", "remove_outliers", "handle_public_data", "validate_public_records", "create_derived_features"],
        "tourism": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_location_data", "standardize_tourism_codes", "encode_categorical", "remove_outliers", "handle_travel_data", "validate_tourism_metrics", "create_derived_features"],
        "hospitality": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_location_data", "standardize_hotel_codes", "encode_categorical", "remove_outliers", "handle_booking_data", "validate_hospitality_metrics", "create_derived_features"],
        "automotive": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_vehicle_codes", "standardize_manufacturer_names", "encode_categorical", "remove_outliers", "handle_automotive_data", "validate_vehicle_metrics", "create_derived_features"],
        "technology": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_ip_addresses", "standardize_tech_codes", "encode_categorical", "remove_outliers", "handle_technical_data", "validate_technology_metrics", "create_derived_features"],
        "IT": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_ip_addresses", "standardize_IT_codes", "encode_categorical", "remove_outliers", "handle_system_data", "validate_IT_metrics", "create_derived_features"],
        "software": ["handle_missing_values", "remove_duplicates", "validate_email_format", "validate_ip_addresses", "standardize_software_codes", "encode_categorical", "remove_outliers", "handle_software_metrics", "validate_development_data", "create_derived_features"],
        "biotechnology": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_biological_codes", "standardize_lab_protocols", "encode_categorical", "remove_outliers", "handle_biological_data", "validate_biotech_metrics", "create_derived_features"],
        "bioinformatics": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_genomic_codes", "standardize_sequence_data", "encode_categorical", "remove_outliers", "handle_genomic_data", "validate_bioinformatics_data", "create_derived_features"],
        "social_media": ["handle_missing_values", "remove_duplicates", "validate_urls", "validate_email_format", "standardize_social_platforms", "encode_categorical", "remove_outliers", "handle_engagement_data", "validate_social_metrics", "create_derived_features"],
        "networking": ["handle_missing_values", "remove_duplicates", "validate_ip_addresses", "validate_urls", "standardize_network_codes", "encode_categorical", "remove_outliers", "handle_network_data", "validate_networking_metrics", "create_derived_features"],
        "legal": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_legal_codes", "standardize_case_numbers", "encode_categorical", "remove_outliers", "handle_legal_data", "validate_legal_documents", "create_derived_features"],
        "law": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_legal_codes", "standardize_case_numbers", "encode_categorical", "remove_outliers", "handle_legal_data", "validate_legal_documents", "create_derived_features"],
        "construction": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_construction_codes", "standardize_project_numbers", "encode_categorical", "remove_outliers", "handle_construction_data", "validate_building_metrics", "create_derived_features"],
        "engineering": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_engineering_codes", "standardize_project_numbers", "encode_categorical", "remove_outliers", "handle_engineering_data", "validate_engineering_metrics", "create_derived_features"],
        "aerospace": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_aerospace_codes", "standardize_aircraft_data", "encode_categorical", "remove_outliers", "handle_aerospace_data", "validate_flight_metrics", "create_derived_features"],
        "aviation": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_aviation_codes", "standardize_aircraft_data", "encode_categorical", "remove_outliers", "handle_aviation_data", "validate_flight_metrics", "create_derived_features"],
        "maritime": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_maritime_codes", "standardize_vessel_data", "encode_categorical", "remove_outliers", "handle_maritime_data", "validate_shipping_metrics", "create_derived_features"],
        "naval": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_naval_codes", "standardize_vessel_data", "encode_categorical", "remove_outliers", "handle_naval_data", "validate_naval_metrics", "create_derived_features"],
        "mining": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_mining_codes", "standardize_mineral_data", "encode_categorical", "remove_outliers", "handle_mining_data", "validate_mining_metrics", "create_derived_features"],
        "resources": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_resource_codes", "standardize_resource_data", "encode_categorical", "remove_outliers", "handle_resource_data", "validate_resource_metrics", "create_derived_features"],
        "textiles": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_textile_codes", "standardize_material_data", "encode_categorical", "remove_outliers", "handle_textile_data", "validate_textile_metrics", "create_derived_features"],
        "fashion": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_fashion_codes", "standardize_size_data", "encode_categorical", "remove_outliers", "handle_fashion_data", "validate_fashion_metrics", "create_derived_features"],
        "food_beverage": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_food_codes", "standardize_ingredient_data", "encode_categorical", "remove_outliers", "handle_food_data", "validate_nutrition_metrics", "create_derived_features"],
        "restaurant": ["handle_missing_values", "remove_duplicates", "standardize_date_format", "validate_menu_codes", "standardize_ingredient_data", "encode_categorical", "remove_outliers", "handle_restaurant_data", "validate_restaurant_metrics", "create_derived_features"]
    }
    return guidelines.get(domain, ["handle_missing_values", "remove_duplicates", "fix_data_types", "standardize_format", "validate_ranges", "encode_categorical", "remove_outliers", "create_derived_features"])

def validate_domain_detection(domain_info, df):
    if domain_info['confidence'] == 'low':
        return False
        
    domain_keywords = {
        'sales': ['sale', 'customer', 'revenue', 'product', 'order', 'price'],
        'ecommerce': ['product', 'order', 'customer', 'price', 'cart', 'sku'],
        'finance': ['account', 'transaction', 'balance', 'amount', 'currency', 'financial'],
        'healthcare': ['patient', 'medical', 'diagnosis', 'treatment', 'hospital', 'health'],
        'education': ['student', 'course', 'grade', 'school', 'teacher', 'academic'],
        'real_estate': ['property', 'house', 'price', 'location', 'square', 'estate'],
        'manufacturing': ['product', 'production', 'quality', 'machine', 'assembly', 'manufacture'],
        'logistics': ['shipment', 'delivery', 'tracking', 'warehouse', 'supply', 'logistics']
    }
    
    domain = domain_info['domain']
    if domain not in domain_keywords:
        return True
        
    keywords = domain_keywords[domain]
    column_names = [col.lower() for col in df.columns]
    matches = sum(1 for keyword in keywords if any(keyword in col for col in column_names))
    return matches >= 2