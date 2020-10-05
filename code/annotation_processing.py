import numpy as np
import pandas as pd
import os
import re

# Definitions
AnnotationPath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\crowdskin\\data-clean'
DataTypeFilename = 'data_types.csv'
TrainPath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\ISIC-2017_Training_Part3_GroundTruth.csv'
ValidationPath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\ISIC-2017_Validation_Part3_GroundTruth.csv'
TestPath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\ISIC-2017_Test_v2_Part3_GroundTruth.csv'

# Variables
NumGroups = 10  # Amount of groups
NumImages = 100  # Amount of images per group
AnnotationBalance = 0.3  # Distribution Malignant Benign for images
# Image has to be annotated for these annotation types to be considered annotated. It is possible to choose from:
# ['Asymmetry', 'Border', 'Color', 'Color_Categorised', 'Dermo',
#        'Blood', 'Blue', 'Color_Fade', 'Compactness', 'Erythema', 'Red',
#        'Black', 'Skin_Color', 'Irregular_pigmentation', 'Flaking',
#        'Rough_Surface', 'Irregularity', 'White', 'Diameter', 'Yellow',
#        'Brown']
RequiredAnnotations = ['Asymmetry', 'Border', 'Color']


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


def create_annotation_df(annotation_path, data_type_filename):
    # loops through all folders from each year and applies the select_data to generate one large dataframe containing
    # features.
    clean_annotations_df = pd.DataFrame()

    for subdir, dirs, files in os.walk(annotation_path):
        files_data = [s for s in files if "group" in s]
        data_types_path = os.path.join(subdir, data_type_filename)
        for file in files_data:
            year = subdir[-9:]
            file_path = os.path.join(subdir, file)
            group = int(re.findall("\d+", file)[0])
            clean_annotations = select_data(file_path, data_types_path, year, group)
            clean_annotations_df = clean_annotations_df.append(clean_annotations, ignore_index=True)

    return clean_annotations_df


def get_annotations(annotations_df, ISIC_ID):
    # returns pandas series of all annotations as lists.
    annotation_df = annotations_df[annotations_df['ID'] == ISIC_ID]
    annotation = annotation_df.groupby('data_type')['data'].apply(list)
    return annotation


def load_ground_truth(Train_Path, Validation_Path, Test_Path):
    # loads ground truth as dataframe from ISIC train, validation and test csv's
    GT_raw_validation = pd.read_csv(Validation_Path)
    GT_raw_test = pd.read_csv(Test_Path)
    GT_raw_train = pd.read_csv(Train_Path)
    GT_df = GT_raw_train.append(GT_raw_validation.append(GT_raw_test))
    GT_df = GT_df.astype({'melanoma': 'bool', 'seborrheic_keratosis': 'bool'})
    GT_df['malignant'] = GT_df['melanoma'] | GT_df['seborrheic_keratosis']
    return GT_df


def check_annotations(df_annotations, req_annotations):
    # Checks if images are already annotated based on list req_annotations. (has to be annotated for these datatypes
    # to be counted as an annotation). Then using load_ground_truth creates a dataframes for benign and malignant that
    # are annotated or not annotated
    contains_annotation = df_annotations.groupby(['ID']).apply(lambda x: sum(x['data_type'] == req_annotations[0]) > 0)
    for i in range(len(req_annotations) - 1):
        i += 1
        contains_group = df_annotations.groupby(['ID']).apply(lambda x: sum(x['data_type'] == req_annotations[i]) > 0)
        contains_annotation = contains_annotation & contains_group
    contains_annotation = contains_annotation[contains_annotation == True].index

    GT = load_ground_truth(TrainPath, ValidationPath, TestPath)
    GT_Malignant = GT[GT['malignant'] == True].drop('malignant', axis=1)
    GT_Benign = GT[GT['malignant'] == False].drop('malignant', axis=1)

    ID_malignant = GT_Malignant[GT_Malignant['image_id'].isin(contains_annotation) == False].sort_values('image_id', ignore_index=True)
    ID_benign = GT_Benign[GT_Benign['image_id'].isin(contains_annotation) == False].sort_values('image_id', ignore_index=True)
    ID_malignant_annotated = GT_Malignant[GT_Malignant['image_id'].isin(contains_annotation)].sort_values('image_id', ignore_index=True)
    ID_benign_annotated = GT_Benign[GT_Benign['image_id'].isin(contains_annotation)].sort_values('image_id', ignore_index=True)
    return ID_malignant, ID_benign, ID_malignant_annotated, ID_benign_annotated


def create_group_sets(df_annotations, num_groups, amount_img, class_balance, required_annotations):
    # Devides annotations into dataframes based on the number of groups, the amount of images per group, and the
    # distribution between malignant and benign. Returns a dictionairy containing a dataframe with ISIC images for each
    # group number.
    ID_malignant, ID_benign, ID_malignant_ann, ID_benign_ann = check_annotations(df_annotations, required_annotations)
    ID_malignant = ID_malignant.append(ID_malignant_ann)
    ID_benign = ID_benign.append(ID_benign_ann)
    num_malignant = np.rint(amount_img * class_balance).astype(int)
    num_benign = np.rint(amount_img * (1 - class_balance)).astype(int)
    ann_dict = {}
    for i in range(num_groups):
        i += 1
        df_group_benign = ID_benign.iloc[i * num_benign:i * num_benign + 1]
        df_group_malignant = ID_malignant.iloc[i * num_benign:i * num_malignant + 1]
        df_group = df_group_benign.append(df_group_malignant)
        if i < 10:
            ann_dict['group0' + str(i)] = df_group
        else:
            ann_dict['group' + str(i)] = df_group
    return ann_dict


df = create_annotation_df(AnnotationPath, DataTypeFilename)
test_annotation = get_annotations(df, 'ISIC_0014092')
images_per_group = create_group_sets(df, NumGroups, NumImages, AnnotationBalance, RequiredAnnotations)
