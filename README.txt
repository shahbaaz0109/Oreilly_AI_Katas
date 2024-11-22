Shopwise AI Assistant

Welcome to the Shopwise AI Assistant! This application is designed to help you analyze product and order data using a conversational AI interface. The AI assistant is powered by OpenAI’s GPT-4 model and is capable of performing comprehensive data analysis on the provided datasets.


Features

Conversational Interface: Interact with the AI assistant to get insights from your data.
Comprehensive Data Analysis: The assistant analyzes the complete dataset, ensuring no data is truncated or overlooked.
Customizable: Easily upload your product and order CSV files to start the analysis.
Memory: The assistant maintains context of previous questions and answers, providing a seamless conversational experience


To run the Shopwise AI Assistant, you need to have Python installed on your system. Follow the steps below to set up and run the application:

1. Clone the Repository:

git clone https://github.com/shahbaaz0109/Oreilly_AI_Katas.git
    cd Oreilly_AI_Katas

2. Install Dependencies:
pip install langchain openai pandas streamlit python-dotenv langchain-experimental langchain-openai langchain-core langchain_community --q

3. Run the Application:
streamlit run Oreilly_AI_Assistant.py



Steps to Use the AI Assistant


1. Enter OpenAI API Key:
When you run the application, you will be prompted to enter your OpenAI API Key. This key is required to interact with the OpenAI API.


2. Upload Data:
Use the sidebar to upload your Product CSV and Order CSV files.
Once the files are uploaded, the assistant will load and analyze the data.


3. Interact with the Assistant:
Type your questions in the chat input at the bottom of the page.
The assistant will respond with insights based on the complete dataset.


4. Clear Chat History:
Use the “Clear Chat History” button in the sidebar to reset the conversation.


Troubleshooting

1. API Connection Issues: Ensure your OpenAI API key is correct and has sufficient quota.
2. Data Loading Issues: Verify that the uploaded CSV files are in the correct format and contain the expected columns.

