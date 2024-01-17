import os
from openai import OpenAI
import streamlit as st
import pandas as pd
import logging
import ast
from PyPDF2 import PdfReader
from datetime import datetime
import base64

raw_json = '''
[
    {
        "Bank Information" : {
                                "Bank Name" : "",
                                "Bank Address" : "",
                                "Bank Contact Number" : "",
                                "User Account Number" : "",
                                "User Address" : ""
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

def extract_pdf_content(uploaded_file):
    file_path = uploaded_file.name
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
                    #print("PDF Password Found.")
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
                    #print("Incorrect password. Could not decrypt the PDF.")
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
                #print("File password not found in Database.")
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
                    'user_account_number': response_text[0]["Bank Information"]["User Account Number"],
                    'user_address': response_text[0]["Bank Information"]["User Address"],
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
                    'total_ATM_withdrawals': atm_withdrawls_sum,
                    'total_Card_withdrawals': atm_deposits_sum,
                    'total_Electronic_withdrawals': cheques_withdrawls_sum,
                    'total_Other_withdrawals': cheques_deposits_sum,
                    'total_ATM_deposits': electronic_withdrawls_sum,
                    'total_Card_deposits': electronic_deposits_sum,
                    'total_Electronic_deposits': other_withdrawls_sum,
                    'total_Other_deposits': other_deposits_sum
                }
    dataframe1 = pd.read_excel('transaction_data.xlsx')
    dataframe2 = pd.DataFrame([sample_data])
    dataframe3 = dataframe2.T
    dataframe1 = pd.concat([dataframe1, dataframe2], ignore_index=False)
    dataframe1.to_excel('transaction_data.xlsx', index=False)
    return dataframe3

def handling_gpt_ouput(gpt_response):
    try:
        # Try parsing the variable as a list
        parsed_variable = ast.literal_eval(gpt_response)
        #logging.info('GPT response parsed successfully.')
        if isinstance(parsed_variable, list):
            # If it's already a list, return it as is
            #logging.info('GPT response is already a JSON inside list.')
            return parsed_variable
    except (ValueError, SyntaxError):
        pass
    # Extract content between first and last curly braces
    start_index = gpt_response.find('{')
    end_index = gpt_response.rfind('}')
    try:
        if start_index != -1 and end_index != -1:
            extracted_content = gpt_response[start_index:end_index + 1]
            #logging.info(f'Extracted GPT response as JSON in string format: {extracted_content}')
            output = eval(extracted_content)
            #logging.info(f'Evaluated(eval()) string JSON response inside list: {[output]}')
            return  [output]  # Return the extracted content as a list
        #logging.exception(f'handling_gpt_output_failed()- returning empty list :{gpt_response}')
    except:
        #print('Got error')
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
                Instruction1 is Extract Bank Name, Bank Address, Bank Contact Number, User Account Number, User Address from the top of statement.\n
                Instruction2 is Classify the transaction data month wise and insert the sum of amount into the raw json format in respective keys.\n
                Instruction3 is Do not skip any data in response, you have to write everything in response, do not leave anything for user to follow pattern.\n
                Instruction4 is Return only response in JSON format.\n
                User Bank statement content is: {extracted_statement} \n
                '''
    prompt2=f''' Give me the complete output in Json format from the response do not skip anything'''

    conversation1 = [{'role': 'system', 'content': system},{'role': 'user', 'content': prompt2}]
    response = client.chat.completions.create(model=model_engine,messages=conversation1,temperature = 0)
    jsonify_response = response.choices[0].message.content
    output = handling_gpt_ouput(jsonify_response)
    #logging.info(output)
    return output

# Create a function to handle file download
def download_file(file_content, file_name):
    b64 = base64.b64encode(file_content.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download Transaction Summary</a>'
    return href

def app_layout():
    extracted_data = extract_pdf_content(uploaded_file)
    #print(extracted_data)

    if extracted_data is not None:
        response_text = extract_information_from_text(extracted_data)
        #print('\n\n',response_text)
        output = dataframe(response_text)
        st.table(output)
        generated_file = output.to_csv(index=False)
        st.subheader(' ')
        st.markdown(download_file(generated_file, 'transaction_summary'), unsafe_allow_html=True)
        return  response_text
    else:
        #print('\n\nPDF Extractor returned None.')
        st.write('PDF Extraction Failed!')
        return  None

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

    st.title("Liberis Statement Insights")
    uploaded_file = st.file_uploader("Upload Bank Statement 👇🏻")
    if uploaded_file is None:
        st.write("Waiting for file upload...")
    if uploaded_file is not None:
        button = st.button("Generate Summary 📝")
        if button:
            with st.spinner("Generating"):
                app_layout()
