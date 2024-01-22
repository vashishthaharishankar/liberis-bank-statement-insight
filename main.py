import os
from openai import OpenAI
import streamlit as st
import pandas as pd
import logging
import ast
import re
from PyPDF2 import PdfReader
from datetime import datetime
import base64
from pydantic import BaseModel

if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)


raw_json = '''
[
    {
        "Bank Information" : {
                                "Bank Name" : "",
                                "Bank Address" : "",
                                "Bank Contact Number" : "",
                                "Customer Name" : "",
                                "Customer Account Number" : "",
                                "Customer Address" : ""
                            },

        "Transaction Summary" : [
            {
                "January" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "February" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "March" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "April" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "May" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "June" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "July" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "August" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "September" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "October" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "November" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ],
                "December" : [
                    {
                        "ATM" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Cheques" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Electronic" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ],
                        "Others" : [
                            {
                                "Withdrawls" : "",
                                "Deposits" : ""
                            }
                        ]
                    }
                ]

            }
        ],

        "Overall Bank Statement Insights" : ""
    }
]
'''
def extract_password(file_name):
    '''
    Hard Coded Database: info.xlsx
    '''
    credentials_file_path = 'info.xlsx' # Hard Coded Database
    df = pd.read_excel(credentials_file_path)
    password_list = []
    for index, row in df.iterrows():
        if row['File Name'] == file_name: # Hard Coded Columns Name i.e File Name and Password
            password_list.append(str(row['Password']))
    return password_list  # File not found, return None for both values

class File(BaseModel):
    name: str
    content: bytes

    @classmethod
    def from_bytes(cls, name: str, content: bytes):
        return cls(name=name, content=content)

def extract_pdf_content(uploaded_file):
    # bytes_data = uploaded_file.read() uploaded_files_list.append(File.from_bytes(name=uploaded_file.name, content=bytes_data))
    file_path = uploaded_file.name
    #with open(uploaded_file, 'rb') as pdf_file:
    pdf_reader = PdfReader(uploaded_file)
    # Check if the PDF is encrypted
    print('Is file encrypted? ',pdf_reader.is_encrypted)
    if pdf_reader.is_encrypted:
        # Try to decrypt the PDF with the provided password
        password = extract_password(file_path)
        if password:
            found = False
            for password_name in password:
                if pdf_reader.decrypt(password_name):
                    found = True
                    num_pages = len(pdf_reader.pages)
                    content = ''
                    # Extract text from each page
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text()
                    print("PDF Password Found.")
                    return content
            if not found:
                try:
                    num_pages = len(pdf_reader.pages)
                    content = ''
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text()
                    return content
                except:
                    print("Incorrect password. Could not decrypt the PDF.")
                    st.write("Incorrect password. Could not decrypt the PDF.")
                    return None
        else:
            try:
                num_pages = len(pdf_reader.pages)
                content = ''
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text()
                return content
            except:
                print("File password not found in Database.")
                st.write("File password not found in Database.")
                return None
    else:
        # Extract text from each page if PDF is not encrypted
        num_pages = len(pdf_reader.pages)
        content = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            content += page.extract_text()
        return content

# def generate_insight_based_on_monthly_average(monthly_transactions):
#     client = OpenAI()
#     client.api_key = os.getenv('OPENAI_API_KEY')
#     #model_engine = "gpt-3.5-turbo-1106"
#     model_engine = "gpt-4-1106-preview"
#     system = f'''You are a insight generator of the transactions data, user will provide you four values:\n
#                     1. total_deposits
#                     2. total_withdrawls
#                     3. monthly_average
#                     4. overall_transaction_insights
#                 based on this transaction data you have to generate insight whether should the person qualifies for loan,\n
#                 but before that consider one main point that the threshold to loan amount is 2.5 times of monthly_average\n
#                 example: loan_threshold = 2.5 * monthly_average
#                 Below are the instructions which you have to follow before generating insights:\n
#                 Instruction 1: Display insight depending on the threshold ( calculate threshold using formula 2.5 times average monthly revenue )
#                 about how much lending we should do as per your analysis.\n
#                 Do not show threshold formula to user just consider it only for you.
#                 Instruction 2: Insight should be short. \n
#                 Below is the customer transaction data: {monthly_transactions}
#                 '''
#     prompt2=f''' Return only Insight based on the user transaction data with your analysis. '''

#     conversation1 = [{'role': 'system', 'content': system},{'role': 'user', 'content': prompt2}]
#     response = client.chat.completions.create(model=model_engine,messages=conversation1,temperature = 0)
#     jsonify_response = response.choices[0].message.content
#     logging.info(jsonify_response)
#     return jsonify_response

def average_monthly_balance(filled_gpt_json_data):

    total_deposits = float(filled_gpt_json_data['total_ATM_deposits']) + float(filled_gpt_json_data['total_Cheques_deposits']) + float(filled_gpt_json_data['total_Electronic_deposits']) + float(filled_gpt_json_data['total_Other_deposits'])
    total_withdrawls = float(filled_gpt_json_data['total_ATM_withdrawls']) + float(filled_gpt_json_data['total_Cheques_withdrawls']) + float(filled_gpt_json_data['total_Electronic_withdrawls']) + float(filled_gpt_json_data['total_Other_withdrawls'])
    months_transaction = [
        str(filled_gpt_json_data['january']),str(filled_gpt_json_data['february']),str(filled_gpt_json_data['march']),
        str(filled_gpt_json_data['april']),str(filled_gpt_json_data['may']),str(filled_gpt_json_data['june']),
        str(filled_gpt_json_data['july']),str(filled_gpt_json_data['august']),str(filled_gpt_json_data['september']),
        str(filled_gpt_json_data['october']),str(filled_gpt_json_data['november']),str(filled_gpt_json_data['december'])
        ]
    month_count = 0
    pattern = re.compile(r'[1-9]')
    for month in months_transaction:
        if pattern.search(month):
            month_count += 1
    if month_count>0:
        monthly_average = float(total_deposits)/float(month_count)
    else:
        monthly_average = float(total_deposits)

    # data_for_gpt_to_fetch_lending_insight = {
    #     'total_deposits' : total_deposits,
    #     'total_withdrawls' : total_withdrawls,
    #     'monthly_average' : monthly_average,
    #     'overall_transaction_insights' : filled_gpt_json_data['insights']
    # }
    # insight_based_on_lending_criteria = generate_insight_based_on_monthly_average(data_for_gpt_to_fetch_lending_insight)
    insight_based_on_lending_criteria = f'As per overall transaction analysis, you are eligible for {round(monthly_average*2.5,2)} amount.'
    main_data_shown_to_user = {
                            'Customer Name' : filled_gpt_json_data['customer_name'],
                            'Bank Name' : filled_gpt_json_data['bank_name'],
                            'Address' : filled_gpt_json_data['bank_address'],
                            'Average Monthly Revenue' : monthly_average,
                            'Insight' : insight_based_on_lending_criteria,
                            'Overall transactions insight' : filled_gpt_json_data['insights'],
                        }
    return main_data_shown_to_user

def dataframe(response_text):
    months_name = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    atm_withdrawls = []
    atm_deposits = []
    cheques_withdrawls = []
    cheques_deposits = []
    electronic_withdrawls = []
    electronic_deposits = []
    other_withdrawls = []
    other_deposits = []
    for month in months_name:
        if response_text[0]["Transaction Summary"][0][month]:
            print('\n\n',month,'\n\n')
            print('\n\n',response_text[0]["Transaction Summary"][0][month],'\n\n')
            atm_withdrawls.append(response_text[0]["Transaction Summary"][0][month][0]["ATM"][0]["Withdrawls"])
            atm_deposits.append(response_text[0]["Transaction Summary"][0][month][0]["ATM"][0]["Deposits"])
            cheques_withdrawls.append(response_text[0]["Transaction Summary"][0][month][0]["Cheques"][0]["Withdrawls"])
            cheques_deposits.append(response_text[0]["Transaction Summary"][0][month][0]["Cheques"][0]["Deposits"])
            electronic_withdrawls.append(response_text[0]["Transaction Summary"][0][month][0]["Electronic"][0]["Withdrawls"])
            electronic_deposits.append(response_text[0]["Transaction Summary"][0][month][0]["Electronic"][0]["Deposits"])
            other_withdrawls.append(response_text[0]["Transaction Summary"][0][month][0]["Others"][0]["Withdrawls"])
            other_deposits.append(response_text[0]["Transaction Summary"][0][month][0]["Others"][0]["Deposits"])

    atm_withdrawls_sum = sum([float(x.replace(',','')) for x in atm_withdrawls if x])
    atm_deposits_sum = sum([float(x.replace(',','')) for x in atm_deposits if x])
    cheques_withdrawls_sum = sum([float(x.replace(',','')) for x in cheques_withdrawls if x])
    cheques_deposits_sum = sum([float(x.replace(',','')) for x in cheques_deposits if x])
    electronic_withdrawls_sum = sum([float(x.replace(',','')) for x in electronic_withdrawls if x])
    electronic_deposits_sum = sum([float(x.replace(',','')) for x in electronic_deposits if x])
    other_withdrawls_sum = sum([float(x.replace(',','')) for x in other_withdrawls if x])
    other_deposits_sum = sum([float(x.replace(',','')) for x in other_deposits if x])

    sample_data = {
                    'time_stamp': datetime.now(),
                    'bank_name': response_text[0]["Bank Information"]["Bank Name"],
                    'bank_address': response_text[0]["Bank Information"]["Bank Address"],
                    'bank_contact_number': response_text[0]["Bank Information"]["Bank Contact Number"],
                    'customer_name': response_text[0]["Bank Information"]["Customer Name"],
                    'customer_account_number': response_text[0]["Bank Information"]["Customer Account Number"],
                    'customer_address': response_text[0]["Bank Information"]["Customer Address"],
                    'insights': response_text[0]["Overall Bank Statement Insights"],
                    'january': response_text[0]["Transaction Summary"][0]["January"],
                    'february': response_text[0]["Transaction Summary"][0]["February"],
                    'march': response_text[0]["Transaction Summary"][0]["March"],
                    'april': response_text[0]["Transaction Summary"][0]["April"],
                    'may': response_text[0]["Transaction Summary"][0]["May"],
                    'june': response_text[0]["Transaction Summary"][0]["June"],
                    'july': response_text[0]["Transaction Summary"][0]["July"],
                    'august': response_text[0]["Transaction Summary"][0]["August"],
                    'september': response_text[0]["Transaction Summary"][0]["September"],
                    'october': response_text[0]["Transaction Summary"][0]["October"],
                    'november': response_text[0]["Transaction Summary"][0]["November"],
                    'december': response_text[0]["Transaction Summary"][0]["December"],
                    'total_ATM_withdrawls': atm_withdrawls_sum,
                    'total_ATM_deposits': atm_deposits_sum,
                    'total_Cheques_withdrawls': cheques_withdrawls_sum,
                    'total_Cheques_deposits': cheques_deposits_sum,
                    'total_Electronic_withdrawls': electronic_withdrawls_sum,
                    'total_Electronic_deposits': electronic_deposits_sum,
                    'total_Other_withdrawls': other_withdrawls_sum,
                    'total_Other_deposits': other_deposits_sum
                }
    output_to_display = average_monthly_balance(sample_data)
    dataframe_main = pd.DataFrame([output_to_display])
    dataframe_main = dataframe_main.T

    # dataframe1 = pd.read_excel('transaction_data.xlsx')
    dataframe2 = pd.DataFrame([sample_data])
    dataframe2 = dataframe2.T
    # dataframe1 = pd.concat([dataframe1, dataframe2], ignore_index=False)
    # dataframe1.to_excel('transaction_data.xlsx', index=False)
    return dataframe_main,dataframe2

def handling_gpt_ouput(gpt_response):
    try:
        # Try parsing the variable as a list
        parsed_variable = ast.literal_eval(gpt_response)
        logging.info('GPT response parsed successfully.')
        if isinstance(parsed_variable, list):
            # If it's already a list, return it as is
            logging.info('GPT response is already a JSON inside list.')
            return parsed_variable
    except (ValueError, SyntaxError):
        pass
    # Extract content between first and last curly braces
    start_index = gpt_response.find('{')
    end_index = gpt_response.rfind('}')
    try:
        if start_index != -1 and end_index != -1:
            extracted_content = gpt_response[start_index:end_index + 1]
            logging.info(f'Extracted GPT response as JSON in string format: {extracted_content}')
            output = eval(extracted_content)
            logging.info(f'Evaluated(eval()) string JSON response inside list: {[output]}')
            return  [output]  # Return the extracted content as a list
        logging.exception(f'handling_gpt_output_failed()- returning empty list :{gpt_response}')
    except:
        print('Got error')
        pass
    return []  # Return an empty list if extraction fails

def extract_information_from_text(extracted_statement):
    client = OpenAI()
    client.api_key = os.getenv('OPENAI_API_KEY')
    #model_engine = "gpt-3.5-turbo-1106"
    model_engine = "gpt-4-1106-preview"
    system = f'''You are a Bank statement Auditor, whose task is to fill raw json format keys with the data provided in users bank statement:
                Raw Json Format is: {raw_json} \n
                Before proceding further you have to follow below instructions:
                Instruction 1 is Extract Bank Name, Bank Address, Bank Contact Number, User Account Number, User Address from the top of statement.\n
                Instruction 2 is Classify the transaction data month wise and insert the total sum of all amounts into the raw json format in respective keys.\n
                Instruction 3 is Do not skip any data in response, you have to write everything in response, do not leave anything for user to follow pattern.\n
                Instruction 4 is Fetch insight based on the overall transaction trends and pattern and insert that into overall_insight key of josn.\n
                Instruction 5 is Return only response in JSON format please do not delete any keys of raw , structure should not change of unfilled json.\n
                User Bank statement content is: {extracted_statement} \n
                '''
    prompt2=f''' Give me the complete output in Json format from the response do not skip anything'''

    conversation1 = [{'role': 'system', 'content': system},{'role': 'user', 'content': prompt2}]
    response = client.chat.completions.create(model=model_engine,messages=conversation1,temperature = 0)
    jsonify_response = response.choices[0].message.content
    print(jsonify_response)
    output = handling_gpt_ouput(jsonify_response)
    logging.info(output)
    return output

# Create a function to handle file download
def download_file(file_content, file_name):
    b64 = base64.b64encode(file_content.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">👉🏻 Download detailed report</a>'
    return href
def app_layout():
    extracted_data = extract_pdf_content(uploaded_file)
    print(extracted_data)

    if extracted_data is not None:
        response_text = extract_information_from_text(extracted_data)
        print('\n\n',response_text)
        output = dataframe(response_text)
        st.table(output[0])
        #st.table(output[1])
        generated_file = output[1].to_csv(index=False)
        st.subheader(' ')
        st.markdown(download_file(generated_file, 'Detailed Report'), unsafe_allow_html=True)

        return  response_text
    else:
        print('\n\nPDF Extractor returned None.')
        st.write('\n\nPDF Extraction Failed.')
        return  None

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    output = ()
    st.title("Liberis Statement Insights")
    uploaded_file = st.file_uploader("Upload Bank Statement")
    if uploaded_file is None:
        st.write("Waiting for file upload...")
    if uploaded_file is not None:
        button = st.button("Generate Summary 📝")
        if button:
            with st.spinner("Processing..."):
                app_layout()
