import pandas as pd
import numpy as np
import ast
import re
from BIDS.core.vert_constants import v_name2idx, v_idx2name

# # Specify the file path
# excel_file_path = '/media/DATA/martina_ma/pathologic_fractures_clean.xlsx'

# # Read the Excel file into a pandas DataFrame
# # Read a specific sheet by name


# print(v_idx2name)
# print(v_name2idx)

class Anomaly:
    def __init__(self, df_row):
        self.name = df_row['ID'].lower()
        self.sex = df_row['sex']
        self.label = df_row['label']
        self.age = df_row['age']
        self.fracture_heights = df_row['fracture_heights'].replace(" ", "").split('_')
        self.fracture_heights_mapped= [v_name2idx.get(value, -1) for value in  df_row['fracture_heights'.replace(" ", "")].split('_')]#.map(v_name2idx)
        self.main_fracture = v_name2idx.get(df_row['main_fracture'.replace(" ", "")], -1) #df_row['main_fracture']
        self.exam_date = df_row['exam_date']


def add_pathologic_fractures_clean_data(emb_df, filepath = '/media/DATA/martina_ma/pathologic_fractures_clean.xlsx'):
    #anomaly_dict = m.make_anomaly_dict()
    #print(type(anomaly_dict))
    df = pd.read_excel(filepath, sheet_name='data')

    emb_df["fracture_flag"] = 'NF'
    emb_df["sex"] = 'U'
    emb_df["age"] = -1
    emb_df["fracture_heights"] = None
    emb_df["exam_date"] = -1
    emb_df["cancerous"] = -1
    for index, row in emb_df.iterrows():
        subject = row['subject'].strip("[]").replace("'", "")
        #print(df['ID'])
        if subject not in df['ID'].values:
            continue
        #print('included')
        selected_row = df.loc[df['ID'] == subject].iloc[0]#df['ID'] == subject
        emb_df.at[index,'cancerous'] = selected_row['label']
        emb_df.at[index, 'sex'] = selected_row['sex']
        emb_df.at[index, 'age'] = selected_row['age']
        emb_df.at[index,"exam_date"] = selected_row['exam_date']
        emb_df.at[index, 'fracture_heights'] = [v_name2idx.get(value, -1) for value in  selected_row['fracture_heights'.replace(" ", "")].split('_')]
        label = int(row['label'].strip("[]").replace("'", ""))
        if label == v_name2idx.get(selected_row['main_fracture'.replace(" ", "")], -1):
            emb_df.at[index,"fracture_flag"] = 'F'
        #print(type(anomaly_id))
        #print(anomaly_id)
    #     label = int(row['label'])
    #     a = anomaly_dict.get(anomaly_id, -1)
    #     #print(type(a.main_fracture))
    #     #print(type(label))
    #     if a == -1:
    #         continue
    #     #print(label == a.main_fracture)
    #     if int(label) == a.main_fracture:
    #          emb_df.at[index, 'fracture_flag'] = 'F'
    #          print('fracture detected')
    # df = pd.read_excel(filepath, sheet_name='data')
    # anomaly_dict = {}
    # # Iterate over each row in the DataFrame and create a Person instance
    # for index, row in df.iterrows():
    #     person_id = row['ID'].lower()
    #     if person_id not in anomaly_dict:
    #         anomaly_dict[person_id] = Anomaly(row)
   # print(type(anomaly_dict))
   # print(anomaly_dict)
    return emb_df

def add_med_paper_anomaly_data(emb_df, filepath = '/media/DATA/martina_ma/Med_Paper_Anomaly.xlsx'):
    df = pd.read_excel(filepath, sheet_name='Shape')
    emb_df['CT scanner'] = -1
    for index, row in emb_df.iterrows():
        subject = row['subject'].strip("[]").replace("'", "")
        val = df['ID full'].values
        arr = []
        for s in val:
            #print(type(s))
            if(isinstance(s,float)):
                #print('stop')
                arr.append(False)
                continue
            #print(s in subject)
            included = subject in s
            arr.append(included)
            #continue
       # arr = [subject in s for s in val]
        if not any(arr):
            continue
        matching_rows = df['ID full'][np.array(arr)]
        for subject_full in matching_rows:
        # if subject not in df[0].values:
        #     continue
        #print('included')
        #
            selected_row = df.loc[df['ID full'] == subject_full].iloc[0]#df['ID'] == subject
            # s = selected_row[1]
            # print(subject in df['ID full'])
            # if subject not in df['ID full'].values:
            #     continue
            #print('included')
            #selected_row = df.loc[df['ID full'] == subject].iloc[0]#df['ID'] == subject
            if emb_df.at[index, 'sex'] == 'U':
                emb_df.at[index, 'sex'] = 'm' if selected_row['gender'] == 1 else 'f'
            
            emb_df.at[index, 'CT scanner'] = selected_row['CT scanner']
            if emb_df.at[index, 'age'] == -1:
                emb_df.at[index, 'age'] =selected_row['patient age (years)']
    #if 
    return emb_df


def add_fxclass_bs_anomaly_data(emb_df, filepath = "/media/DATA/martina_ma/fxclass_bs_anomaly.csv"):
    df = pd.read_csv(filepath, delimiter=";", header = None)
    anomaly_dict = {}
    # Iterate over each row in the DataFrame and create a Person instance
    # for index, row in df.iterrows():
    #     person_id = extract_subject(row[0],'fxclass')
    #     if person_id not in anomaly_dict:
    #         anomaly_dict[person_id] = parse_string(row[1])#[v_name2idx.get(value, -1) for value in parse_string(row[1])] 

    emb_df["fracture_grading"] = -1
    emb_df["Implant"] = 'U'
    #anomaly_dict_fx = m.add_fxclass_fracture_anomaly()
    for index, row in emb_df.iterrows():
        subject = row['subject'].strip("[]").replace("'", "")
        val = df[0].values
        arr = [subject in s for s in val]
        if not any(arr):
            continue
        matching_rows = df[0][np.array(arr)]

        #print(subject in val)
        #v = val.str
        #print(val.str.contains(subject))
         # Check if subject is a substring in any of the strings in the first column of df
        # if not any(df[0].str.contains(subject)):
        #     continue
        for subject_full in matching_rows:
        # if subject not in df[0].values:
        #     continue
        #print('included')
        #
            selected_row = df.loc[df[0] == subject_full].iloc[0]#df['ID'] == subject
            s = selected_row[1]
            #s = selected_row[2]
            #dict = s[1]
            fractures_dict = parse_string(selected_row[1])
            #print(type(anomaly_id))
            #print(anomaly_id)
            label = v_idx2name.get(int(row['label'].strip("[]").replace("'", "")),-1)
            #fractures_dict = anomaly_dict_fx.get(anomaly_id, -1)
            #print(type(a.main_fracture))
            #print(type(label))
            if fractures_dict == []:
                continue
            print("fracture dict exists")
            value = fractures_dict.get(label,-1)
            if value == -1:
                continue
            if 'F' in value:
                emb_df.at[index,"fracture_flag"] = 'F'
                emb_df.at[index,"fracture_grading"] = int(re.search(r'\d+', value).group())
            elif 'Metal' in value:
                emb_df.at[index,"Implant"] = 'M'

    return emb_df

def add_ctfu_bs_anomaly_data(emb_df, filepath = "/media/DATA/martina_ma/ctfu_bs_anomaly.csv"):
    df = pd.read_csv(filepath, delimiter=";", header= None)
    anomaly_dict = {}
    # Iterate over each row in the DataFrame and create a Person instance
    # for index, row in df.iterrows():
    #     person_id = extract_subject(row[0],'ctfu')
    #     if person_id not in anomaly_dict:
    #         anomaly_dict[person_id] = parse_string(row[1])#[v_name2idx.get(value, -1) for value in parse_string(row[1])] 

    #emb_df["fracture_grading"] = -1
    #emb_df["Implant"] = 'NM'
    #anomaly_dict_fx = m.add_fxclass_fracture_anomaly()
    for index, row in emb_df.iterrows():
        subject = row['subject'].strip("[]").replace("'", "")
        val = df[0].values
        arr = [subject in s for s in val]
        if not any(arr):
            continue
        matching_rows = df[0][np.array(arr)]
        # if not any(df[0].str.contains(subject)):
        #     continue
        
        # if subject not in df[0].values:
        #     continue
        #print('included')
        for subject_full in matching_rows:
        # if subject not in df[0].values:
        #     continue
        #print('included')
        #
            selected_row = df.loc[df[0] == subject_full].iloc[0]#df['ID'] == subject
            fractures_dict = parse_string(selected_row[1])
            #print(type(anomaly_id))
            #print(anomaly_id)
            label = v_idx2name.get(int(row['label'].strip("[]").replace("'", "")),-1)
            #fractures_dict = anomaly_dict_fx.get(anomaly_id, -1)
            #print(type(a.main_fracture))
            #print(type(label))
            if fractures_dict == -1:
                continue
            print("fracture dict exists")
            value = fractures_dict.get(label,-1)
            if value == -1:
                continue
            if 'F' in value:
                emb_df.at[index,"fracture_flag"] = 'F'
                emb_df.at[index,"fracture_grading"] = int(re.search(r'\d+', value).group())
            elif 'Metal' in value:
                emb_df.at[index,"Implant"] = 'M'
    return emb_df


def extract_subject(input_string, sub):
    # Define a regular expression pattern to match "fxclassXYZ"
    pattern = re.compile(sub+r'\d+')

    # Use search to find the first match in the input string
    match = pattern.search(input_string)

    # Return the matched string or None if no match is found
    return match.group() if match else None

def extract_ctfu(input_string):
    # Define a regular expression pattern to match "fxclassXYZ"
    pattern = re.compile(r'ctfu\d+')

    # Use search to find the first match in the input string
    match = pattern.search(input_string)

    # Return the matched string or None if no match is found
    return match.group() if match else None


def parse_string(s):
    try:
        # Safely evaluate the string as a dictionary
        dictionary = ast.literal_eval(s)

        # Extract and return the keys as a list
        keys_list = list(dictionary.keys())
        return dictionary #keys_list
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing the input string: {e}")
        return []