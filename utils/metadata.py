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

# class Anomaly:
#     def __init__(self, df_row):
#         self.name = df_row['ID'].lower()
#         self.sex = df_row['sex']
#         self.label = df_row['label']
#         self.age = df_row['age']
#         self.fracture_heights = df_row['fracture_heights'].replace(" ", "").split('_')
#         self.fracture_heights_mapped= [v_name2idx.get(value, -1) for value in  df_row['fracture_heights'.replace(" ", "")].split('_')]#.map(v_name2idx)
#         self.main_fracture = v_name2idx.get(df_row['main_fracture'.replace(" ", "")], -1) #df_row['main_fracture']
#         self.exam_date = df_row['exam_date']

def add_labels(emb_df):
    emb_df["fracture_flag"] = 'U'
    emb_df["sex"] = 'U'
    emb_df["age"] = -1
    emb_df["fracture_heights"] = None
    emb_df["exam_date"] = -1
    emb_df["cancerous"] = -1
    emb_df['CT scanner'] = -1
    emb_df["fracture_grading"] = -1
    emb_df["Implant"] = 'U'
    extended_df = add_pathologic_fractures_clean_data(emb_df)
    extended_df = add_med_paper_anomaly_data(extended_df)
    extended_df = add_fxclass_bs_anomaly_data(extended_df)
    extended_df = add_ctfu_bs_anomaly_data(extended_df)
    return extended_df

def add_pathologic_fractures_clean_data(emb_df, filepath = '/media/DATA/martina_ma/pathologic_fractures_clean.xlsx'):
    df = pd.read_excel(filepath, sheet_name='data')

    # emb_df["fracture_flag"] = 'U'
    # emb_df["sex"] = 'U'
    # emb_df["age"] = -1
    # emb_df["fracture_heights"] = None
    # emb_df["exam_date"] = -1
    # emb_df["cancerous"] = -1
    for index, row in emb_df.iterrows():
        subject = row['subject']#.strip("[]").replace("'", "")
        # if "222" in subject:
        #     if int(row['label']) == 19:
        #         print('stop')
        #print(df['ID'])
        # if "305" in subject:
        #     print('stop')
        if subject not in df['ID'].values:
            continue
        #print('included')
        selected_row = df.loc[df['ID'] == subject].iloc[0]#df['ID'] == subject
        emb_df.at[index,'cancerous'] = selected_row['label']
        emb_df.at[index, 'sex'] = selected_row['sex']
        emb_df.at[index, 'age'] = selected_row['age']
        emb_df.at[index,"exam_date"] = selected_row['exam_date']
        fh = [value for value in  selected_row['fracture_heights'].split('_')]#.replace(" ", "")]
        emb_df.at[index, 'fracture_heights'] = [v_name2idx.get(f.replace(" ", ""), -1) for f in fh]
        x = emb_df.at[index, 'fracture_heights']
        label = int(row['label'])#.strip("[]").replace("'", ""))
        mf = selected_row['main_fracture'].replace(" ", "")
        if label == v_name2idx.get(mf, -1): #evtl replace wieder hinter main_fra
            emb_df.at[index,"fracture_flag"] = 'F'
            emb_df.at[index,"fracture_grading"] = 4
        elif label in x:#emb_df['fracture_heights']:
            emb_df.at[index,"fracture_flag"] = 'F'
            emb_df.at[index,"fracture_grading"] = 4
        else:
            emb_df.at[index,"fracture_flag"] = 'NF'
            emb_df.at[index,"fracture_grading"] = 0
    return emb_df

def add_med_paper_anomaly_data(emb_df, filepath = '/media/DATA/martina_ma/Med_Paper_Anomaly.xlsx'):
    df = pd.read_excel(filepath, sheet_name='Shape')
    #emb_df['CT scanner'] = -1
    for index, row in emb_df.iterrows():
        subject = row['subject']#.strip("[]").replace("'", "")
        val = df['ID full'].values
        arr = []
        for s in val:
            if(isinstance(s,float)):
                arr.append(False)
                continue
            included = subject in s
            arr.append(included)
        if not any(arr):
            continue
        matching_rows = df['ID full'][np.array(arr)]
        for subject_full in matching_rows:
            selected_row = df.loc[df['ID full'] == subject_full].iloc[0]
            if emb_df.at[index, 'sex'] == 'U':
                emb_df.at[index, 'sex'] = 'm' if selected_row['gender'] == 1 else 'f'
            
            emb_df.at[index, 'CT scanner'] = selected_row['CT scanner']
            if emb_df.at[index, 'age'] == -1:
                emb_df.at[index, 'age'] =selected_row['patient age (years)'] 
    return emb_df


def add_fxclass_bs_anomaly_data(emb_df, filepath = "/media/DATA/martina_ma/fxclass_bs_anomaly_cleaned.csv"):
    df = pd.read_csv(filepath, delimiter=";", header = None)
    # emb_df["fracture_grading"] = -1
    # emb_df["Implant"] = 'U'
    for index, row in emb_df.iterrows():
        
        subject = row['subject']#.strip("[]").replace("'", "")
        # if "222" in subject:
        #     if int(row['label']) == 19:
        #         print('stop')
        val = df[0].values
        arr = [subject in s for s in val]
        if not any(arr):
            continue
        matching_rows = df[0][np.array(arr)]
        for subject_full in matching_rows:
            selected_row = df.loc[df[0] == subject_full].iloc[0]
            fractures_dict = parse_string(selected_row[1])
            #print(type(anomaly_id))
            #print(anomaly_id)
            label = v_idx2name.get(int(row['label']),-1)#.strip("[]").replace("'", "")
            if len(fractures_dict) == 0 and row.fracture_flag != 'F' :
                emb_df.at[index,"fracture_flag"] = 'NF'
                emb_df.at[index,"Implant"] = 'NM'
                emb_df.at[index,"fracture_grading"] = 0
                print("no fractures")
                continue
            if len(fractures_dict) == 0 and row.fracture_flag == 'F' :
                continue
            print("fracture dict exists")
            value = fractures_dict.get(label,-1)
            if value == -1:
                if row.fracture_flag == 'F':
                    emb_df.at[index,"Implant"] = 'NM'
                    emb_df.at[index,"fracture_grading"] = 4
                else:
                    emb_df.at[index,"fracture_flag"] = 'NF'
                    emb_df.at[index,"Implant"] = 'NM'
                    emb_df.at[index,"fracture_grading"] = 0
                continue
            if 'F' in value:
                emb_df.at[index,"fracture_flag"] = 'F'
                emb_df.at[index,"fracture_grading"] = int(re.search(r'\d+', value).group())
                emb_df.at[index,"Implant"] = 'NM'
            elif 'Metal' in value:
                emb_df.at[index,"Implant"] = 'M'
                emb_df.at[index,"fracture_flag"] = 'NF' #because metal implant
                emb_df.at[index,"fracture_grading"] = 0

    return emb_df

def add_ctfu_bs_anomaly_data(emb_df, filepath = "/media/DATA/martina_ma/ctfu_bs_anomaly.csv"):
    df = pd.read_csv(filepath, delimiter=";", header= None)
    for index, row in emb_df.iterrows():
        subject = row['subject'].strip("[]").replace("'", "")
        val = df[0].values
        arr = [subject in s for s in val]
        if not any(arr):
            continue
        matching_rows = df[0][np.array(arr)]
        for subject_full in matching_rows:
            selected_row = df.loc[df[0] == subject_full].iloc[0]#df['ID'] == subject
            fractures_dict = parse_string(selected_row[1])
            if fractures_dict == [] and row.fracture_flag != 'F' :
                emb_df.at[index,"fracture_flag"] = 'NF'
                emb_df.at[index,"Implant"] = 'NM'
                emb_df.at[index,"fracture_grading"] = 0
                continue
            if len(fractures_dict) == 0 and row.fracture_flag == 'F' :
                continue
            print("fracture dict exists")
            label = v_idx2name.get(int(row['label']),-1) #.strip("[]").replace("'", "")
            if fractures_dict == -1:
                emb_df.at[index,"fracture_flag"] = 'NF'
                emb_df.at[index,"Implant"] = 'NM'
                emb_df.at[index,"fracture_grading"] = 0
                continue
            print("fracture dict exists")
            value = fractures_dict.get(label,-1)
            if value == -1:
                if row.fracture_flag == 'F':
                    emb_df.at[index,"Implant"] = 'NM'
                    emb_df.at[index,"fracture_grading"] = 4
                else:
                    emb_df.at[index,"fracture_flag"] = 'NF'
                    emb_df.at[index,"Implant"] = 'NM'
                    emb_df.at[index,"fracture_grading"] = 0
                continue
            if 'F' in value:
                emb_df.at[index,"fracture_flag"] = 'F'
                emb_df.at[index,"fracture_grading"] = int(re.search(r'\d+', value).group())
                emb_df.at[index,"Implant"] = 'NM'
            elif 'Metal' in value:
                emb_df.at[index,"Implant"] = 'M'
                emb_df.at[index,"fracture_flag"] = 'NF'
                emb_df.at[index,"fracture_grading"] = 0

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