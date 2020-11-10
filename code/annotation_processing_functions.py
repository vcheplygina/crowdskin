import numpy as np
import pandas as pd
import os
import re


def select_data(data_path, data_type_path, year, group):
    """For a given file convert every value to new dataframe with columns:
    ['ID', 'group_number', 'year', 'annotator', 'orig_column', 'data_type', 'data']"""

    # Read cleaned student files and datatype files
    annotations_df = pd.DataFrame()
    group_df = pd.read_csv(data_path, delimiter=';').dropna(axis=0, how='all').dropna(axis=1, how='all')
    type_df = pd.read_csv(data_type_path, delimiter=';').dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Initiate arrays
    df_length = group_df.shape[0]
    IDs = group_df.iloc[:, 0]
    group_array = np.full(df_length, group)
    year_array = np.full(df_length, year)

    # Select annotations from file and generate a dataframe from the filled arrays
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
    """loops through all folders from each year and applies select_data to generate one large dataframe containing
    features."""

    clean_annotations_df = pd.DataFrame()

    # Walk trough folders containing cleaned annotations
    for subdir, dirs, files in os.walk(annotation_path):
        files_data = [s for s in files if "group" in s]
        data_types_path = os.path.join(subdir, data_type_filename)
        # Select cleaned data using select_data for each annotation file
        for file in files_data:
            year = subdir[-9:]
            file_path = os.path.join(subdir, file)
            group = int(re.findall("\d+", file)[0])
            clean_annotations = select_data(file_path, data_types_path, year, group)
            clean_annotations_df = clean_annotations_df.append(clean_annotations, ignore_index=True)
    return clean_annotations_df


def get_annotations(annotations_df, ISIC_ID):
    """returns pandas series of all annotations as lists."""

    annotation_df = annotations_df[annotations_df['ID'] == ISIC_ID]
    annotation = annotation_df.groupby('data_type')['data'].apply(list)
    return annotation


def load_ground_truth(train_path, validation_path, test_path, malignant_lesion):
    """loads ground truth as dataframe from ISIC train, validation and test csv's"""

    # Read ISIC ground truth data drom csv
    GT_raw_validation = pd.read_csv(validation_path)
    GT_raw_test = pd.read_csv(test_path)
    GT_raw_train = pd.read_csv(train_path)
    GT_df = GT_raw_train.append(GT_raw_validation.append(GT_raw_test))
    GT_df = GT_df.astype({'melanoma': 'bool', 'seborrheic_keratosis': 'bool'})

    # Generate a dataframe with ISIC ID's and if the lesions is considerd malignant (can be chosen by user)
    if malignant_lesion == 'both':
        GT_df['malignant'] = GT_df['melanoma'] | GT_df['seborrheic_keratosis']
    elif malignant_lesion == 'keratosis':
        GT_df['malignant'] = GT_df['seborrheic_keratosis']
    elif malignant_lesion == 'melanoma':
        GT_df['malignant'] = GT_df['melanoma']
    else:
        raise Exception("Choose between 'both', 'keratosis' or 'melanoma' for malignant_lesion")
    return GT_df


def annotation_stats(df_annotations):
    """Prints a few statistics for the dataframe containing all annotations"""

    # Group dataframe by data_type and ID
    df_annotations_grouped = df_annotations.groupby('data_type')
    df_annotations_grouped_id = df_annotations.groupby(['data_type', 'ID'])

    # Calculate ammount, mean and minimum of annotations per category
    stats_df = pd.DataFrame(df_annotations_grouped['data'].count()).rename({'data': 'ann_count'}, axis=1)
    stats_df['mean_ann_img'] = df_annotations_grouped_id['data'].count().mean(level=0)
    stats_df['min_ann_img'] = df_annotations_grouped_id['data'].count().min(level=0)

    print('Annotation statistics: \n')
    print(stats_df)


def drop_annotation_count_categories(df, categories, count):
    """Checks which categories and amounts of annotations should be available before the image is considered annotated.
    Returns an array of ID's that are considered annotated"""

    # Group dataframe by data_type
    grouped_df = df.groupby(['ID', 'data_type'])
    grouped_count = grouped_df['data_type'].count()
    cat_index = grouped_count.index.get_level_values(level=1)

    # Keep ISIC id's that contain the categories and amounts of images requested by the user
    ID = grouped_count.index.get_level_values(level=0)
    for i in range(0, len(categories)):
        grouped_df_cat = grouped_count[cat_index == categories[i]]
        ID_min_count = grouped_df_cat.index.get_level_values(level=0)[grouped_df_cat >= count[i]]
        ID = ID[ID.isin(ID_min_count)].unique()
    return ID


def categorise_annotations(image_ids, train_path, validation_path, test_path, malignant_lesion):
    """ Checks if images are already annotated based on list req_annotations. (has to be annotated for these datatypes
    to be counted as an annotation). Then using load_ground_truth creates dataframes for benign and malignant that
    are annotated or not annotated """

    # Read ground truth
    GT = load_ground_truth(train_path, validation_path, test_path, malignant_lesion)

    # Divide ground truth in malignant and benign
    GT_Malignant = GT[GT['malignant'] == True].drop('malignant', axis=1)
    GT_Benign = GT[GT['malignant'] == False].drop('malignant', axis=1)

    # Divide annotated and not annotated data into malignant and benign dataframe
    ID_malignant = GT_Malignant[GT_Malignant['image_id'].isin(image_ids) == False].sort_values('image_id',
                                                                                               ignore_index=True)
    ID_benign = GT_Benign[GT_Benign['image_id'].isin(image_ids) == False].sort_values('image_id',
                                                                                      ignore_index=True)
    ID_malignant_annotated = GT_Malignant[GT_Malignant['image_id'].isin(image_ids)].sort_values('image_id',
                                                                                                ignore_index=True)
    ID_benign_annotated = GT_Benign[GT_Benign['image_id'].isin(image_ids)].sort_values('image_id',
                                                                                       ignore_index=True)
    return ID_malignant, ID_benign, ID_malignant_annotated, ID_benign_annotated


def create_group_sets(ID_malignant, ID_benign, ID_malignant_ann, ID_benign_ann, num_groups, amount_img, class_balance):
    """Devides annotations into dataframes based on the number of groups, the amount of images per group, and the
    distribution between malignant and benign. Returns a dictionairy containing a dataframe with ISIC images for each
    group number."""

    # Generate malignant and benign dataframes where unannotated images are the first rows
    ID_malignant = ID_malignant.append(ID_malignant_ann)
    ID_benign = ID_benign.append(ID_benign_ann)

    # Calculate the number of benign and malignant images using the ratio provided by the user
    num_malignant = np.rint(amount_img * class_balance).astype(int)
    num_benign = np.rint(amount_img * (1 - class_balance)).astype(int)

    # Generate a dict where every group is assigned a dataframe containing ISIC id's. These dataframes contain the
    # amount of images per group according to the requested malignant/benign ratio
    ann_dict = {}
    for i in range(num_groups):
        i += 1
        df_group_benign = ID_benign.iloc[i * num_benign:(i + 1) * num_benign]
        df_group_malignant = ID_malignant.iloc[i * num_malignant:(i + 1) * num_malignant]
        df_group = df_group_benign.append(df_group_malignant).reset_index(drop=True)
        if i < 10:
            ann_dict['class2020_group0' + str(i)] = df_group
        else:
            ann_dict['class2020_group' + str(i)] = df_group
    return ann_dict


def save_group_sets(ID_group, save_folder):
    """Saves images per group as csv with ground truth and without in assigned folder"""

    # Make folder if it does not exist yet
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save images per group
    for group in ID_group:
        GT_save_name = os.path.join(save_folder, group + '_id.csv')
        GT_name = group + '_labels.csv'
        save_name = os.path.join(save_folder, GT_name)
        ID_group[group].to_csv(save_name, index=False)
        ID_group[group].drop(['melanoma', 'seborrheic_keratosis'], axis=1).to_csv(GT_save_name, index=False)
