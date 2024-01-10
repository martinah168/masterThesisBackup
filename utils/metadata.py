import pandas as pd
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


def make_anomaly_dict(filepath = '/media/DATA/martina_ma/pathologic_fractures_clean.xlsx'):
    df = pd.read_excel(filepath, sheet_name='data')
    anomaly_dict = {}
    # Iterate over each row in the DataFrame and create a Person instance
    for index, row in df.iterrows():
        person_id = row['ID'].lower()
        if person_id not in anomaly_dict:
            anomaly_dict[person_id] = Anomaly(row)
   # print(type(anomaly_dict))
   # print(anomaly_dict)
    return anomaly_dict


def add_fxclass_fracture_anomaly(filepath = "/media/DATA/martina_ma/fxclass_bs_anomaly.csv"):
    df = pd.read_csv(filepath, delimiter=";")
    anomaly_dict = {}
    # Iterate over each row in the DataFrame and create a Person instance
    for index, row in df.iterrows():
        person_id = extract_fxclass(row[0])
        if person_id not in anomaly_dict:
            anomaly_dict[person_id] = [v_name2idx.get(value, -1) for value in parse_string(row[1])] 
    return anomaly_dict


def extract_fxclass(input_string):
    # Define a regular expression pattern to match "fxclassXYZ"
    pattern = re.compile(r'fxclass\d+')

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
        return keys_list
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing the input string: {e}")
        return []