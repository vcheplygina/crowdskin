from annotation_processing_functions import create_annotation_df, get_annotations, create_group_sets, annotation_stats
from annotation_processing_functions import drop_annotation_count_categories, categorise_annotations, save_group_sets
import os

# Paths
crowdskin_path = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\crowdskin'
annotation_path = os.path.join(crowdskin_path, 'data-clean')
save_path = os.path.join(crowdskin_path, '2020-2021')
data_type_filename = 'data_types.csv'
GT_folder = os.path.join(crowdskin_path, 'ISIC_2017_GT')
train_path = os.path.join(GT_folder, 'ISIC-2017_Training_Part3_GroundTruth.csv')
validation_path = os.path.join(GT_folder, 'ISIC-2017_Validation_Part3_GroundTruth.csv')
test_path = os.path.join(GT_folder, 'ISIC-2017_Test_v2_Part3_GroundTruth.csv')

# Variables
num_groups = 10  # Amount of groups
num_img = 100  # Amount of images per group
annotation_balance = 0.3  # Distribution Malignant Benign for images
# Image has to be annotated for these annotation types to be considered annotated. It is possible to choose from:
# ['Asymmetry', 'Border', 'Color', 'Color_Categorised', 'Dermo',
#        'Blood', 'Blue', 'Color_Fade', 'Compactness', 'Erythema', 'Red',
#        'Black', 'Skin_Color', 'Irregular_pigmentation', 'Flaking',
#        'Rough_Surface', 'Irregularity', 'White', 'Diameter', 'Yellow',
#        'Brown']
required_annotations = ['Asymmetry', 'Border', 'Color']
required_amounts = [3, 3, 3]
malignant_lesion = 'melanoma'  # choose between 'both', 'keratosis' or 'melanoma'


df = create_annotation_df(annotation_path, data_type_filename)
annotation_stats(df)

ID_annotated = drop_annotation_count_categories(df, required_annotations, required_amounts)
ID_malignant, ID_benign, ID_malignant_annotated, ID_benign_annotated = categorise_annotations(ID_annotated, train_path, validation_path, test_path, malignant_lesion)
images_per_group = create_group_sets(ID_malignant, ID_benign, ID_malignant_annotated, ID_benign_annotated, num_groups, num_img, annotation_balance)
# save_group_sets(images_per_group, save_path)

