from annotation_processing_functions import create_annotation_df, get_annotations, create_group_sets, annotation_stats
from annotation_processing_functions import drop_annotation_count_categories, categorise_annotations, save_group_sets
# Paths
AnnotationPath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\crowdskin\\data-clean'
SavePath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\OGO beeldverwerking dataset\\crowdskin\\2020-2021'
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
RequiredAmounts = [3, 3, 3]
malignant_lesion = 'melanoma'  # choose between 'both', 'keratosis' or 'melanoma'


df = create_annotation_df(AnnotationPath, DataTypeFilename)
annotation_stats(df)

ID_annotated = drop_annotation_count_categories(df, RequiredAnnotations, RequiredAmounts)
ID_malignant, ID_benign, ID_malignant_annotated, ID_benign_annotated = categorise_annotations(ID_annotated, TrainPath, ValidationPath, TestPath, malignant_lesion)
images_per_group = create_group_sets(ID_malignant, ID_benign, ID_malignant_annotated, ID_benign_annotated, NumGroups, NumImages, AnnotationBalance)
# save_group_sets(images_per_group, SavePath)

