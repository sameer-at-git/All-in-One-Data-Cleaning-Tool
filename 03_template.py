import os

def create_project_structure():
    directories = []
    
    files = {
        "app.py": "",
        "config.py": "",
        "domain_detector.py": "",
        "plan_generator.py": "",
        "data_cleaner.py": "",
        "utils.py": "",
        "requirements.txt": "streamlit\npandas\nnumpy\nlangchain-groq\npython-dotenv",
        "__init__.py": ""
    }
    
    for file_path, content in files.items():
        with open(file_path, 'w') as f:
            f.write(content)
    
    return files.keys()

if __name__ == "__main__":
    created_files = create_project_structure()
    print("Project structure created with files:")
    for file in created_files:
        print(f" - {file}")