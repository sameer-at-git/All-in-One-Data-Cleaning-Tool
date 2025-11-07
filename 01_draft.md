# Data Cleaning App/ UI using Streamlit

The main objective is to take any user input in .csv format that can be any type of dataset such as - users dataset, sales dataset, weather records dataset - basically any domain's any sorta dataset , and providing a csv which is cleaned meaning performing a eda,and other cleaning utilities .

## Flow

- User uploads csv
- program reads the csv
- extracts column info and other info
- extracts what domain is it from like sales, users or any other , it can be infinite number of things
- program uses llm to generate certain eda/cleaning actions as per domain
- program shows actions plan to user as a message and ask - are you okay with these actions ,feel free to add/subtract
- as per the user answers
- perform eda ,get stats on it
- clean data according to traditional domain practices
- provides cleaned data in a csv format for user to download and also shows in a dataframe view in st

## Constraints

- build logic to get domain info from column names or data types or a few rows of data
- llm agent-1 : from domain info,dataset info, generate a plan of actions to perform on the data
- llm agent-2 : finalizing the plan for each actions on each columns
- make it scalable so that huge amount of data and small data can both have same time for generating the output
- if users uploads a clean data llm agent-1: must say your data is already refined are you sure to clean it further
- llm agent-1: will show as much as info as it can to the user based on iniital eda done by program like row number, columnnames, current nul values, etc

## Tech Stack (for now)

- Python , pandas, numpy
- Streamlit
- Langchain , langchain-groq
