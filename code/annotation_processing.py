import numpy as np
import pandas as pd
import os
import re

# Definitions
AnnotationPath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\crowdskin\\data-clean'
DataTypeFilename = 'data_types.csv'


def select_data(DataPath, DataTypePath, year, group):
    # For a given file convert every value to new dataframe with columns:
    # ['ID', 'group_number', 'year', 'annotator', 'orig_column', 'data_type', 'data']

    annotations_df = pd.DataFrame()
    group_df = pd.read_csv(DataPath, delimiter=';').dropna(axis=0, how='all').dropna(axis=1, how='all')
    type_df = pd.read_csv(DataTypePath, delimiter=';').dropna(axis=0, how='all').dropna(axis=1, how='all')

    df_length = group_df.shape[0]
    IDs = group_df.iloc[:, 0]
    group_array = np.full(df_length, group)
    year_array = np.full(df_length, year)
    for i in range(1, group_df.shape[1]):
        column_data = group_df.iloc[:, i]
        data_type = type_df.loc[type_df['group_num'] == group].iloc[:, i]
        column_name = np.full(df_length, column_data.name)
        type_array = np.full(df_length, data_type)
        annotator_array = np.full(df_length, int(column_data.name[-1:]))
        data = dict(ID=IDs, group_number=group_array, year=year_array, annotator=annotator_array,
                    orig_column=column_name, data_type=type_array, data=column_data)
        column_df = pd.DataFrame(data=data).dropna()
        annotations_df = annotations_df.append(column_df, ignore_index=True)
    return annotations_df


clean_annotations_df = pd.DataFrame()

for subdir, dirs, files in os.walk(AnnotationPath):
    files_data = [s for s in files if "group" in s]
    data_types_path = os.path.join(subdir, DataTypeFilename)
    for file in files_data:
        year = subdir[-9:]
        file_path = os.path.join(subdir, file)
        group = int(re.findall("\d+", file)[0])
        clean_annotations = select_data(file_path, data_types_path, year, group)
        clean_annotations_df = clean_annotations_df.append(clean_annotations, ignore_index=True)
