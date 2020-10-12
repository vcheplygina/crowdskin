This code processes annotations for ISIC 2017 lesion images from previous years, and assigns unannotated images to the new groups.
annotation_processing_script.py calls all the functions that are neccesary to assign the images to the new groups. The following
variables need to be set by the user:

# Paths 

AnnotationPath: path to folder containing folders for each year. these folders containin the annotations as csv files.

SavePath: Path to new folder where 2 csv's for each group will be saved. 

DataTypeFilename: Filename of csv where data/annotation types are assigned.

TrainPath: Path to ISIC-2017 part 3 training set ground truth

ValidationPath: Path to ISIC-2017 part 3 validation set ground truth

TestPath: Path to ISIC-2017 part 3 test set ground truth


# Variables

NumGroups: Number of groups that take the course.

NumImages: Number of images that need to be assigned to a group.

AnnotationBalance: Distribution Malignant/Benign for images. 

RequiredAnnotations: List of which annotation types an image has to be annotated before the program considers the image annotated.

RequiredAmounts: List of same length as RequiredAnnotations containing the amount of annotations for each catergory in RequiredAnnotations
                 before the program considers the image annotated.
                 
## Annotation types

The following data/annotation types can be chosen in RequiredAnnotations. Only Asymmetry, Border and Color contain a significant amount of annotations (around 10000 each)

Asymmetry: How asymmetric a lesion is scored on a scale or boolean

Border: How irregular the border of a lesion is scored on a scale or boolean

Color: How uneven the color of a lesion is scored on a scale or boolean

Color_Categorised: Colors that are found in the lesion are noted with comma's between them. (different groups use different methods)

Dermo: Dermascopic/Differential structures. for instance pigment dots or globules.

Blood: Blood vessels can be found in the lesion

Blue: The lesions have a blue glow.

Color_Fade: Lesions have a faded border.

Compactness: uses formula c = circumference^2/(4π*Area)

Erythema: redness surrounding the lesion.

Red: Red is present in the lesion.

Black: Black is present in the lesion.

Skin_Color: Skin_Color is present in the lesion.

Flaking: Lesions shows flaking skin

Rough_Surface: surface of lesion is rough

White: White is present in the lesion.

Diameter: score based on the diameter of the lesion

Yellow: Yellow is present in the lesion.

Brown: Brown is present in the lesion.
              

# Functions

create_annotation_df:  loops through all folders from each year and applies select_data to generate one large dataframe containing
    features. Has columns ['ID', 'group_number', 'year', 'annotator', 'orig_column', 'data_type', 'data']. This dataframe can be 
    used to select data for more specific applications.

annotation_stats: Uses annotations_df to print the ammount of annotations (ann_count), average amount of annotations per image(mean_ann_img) 
and minimum amount of annotations per image(min_ann_img) for each annotation type

get_annotations: returns pandas series of all annotations as lists using annotations_df and an ISIC_ID as a string. 

drop_annotation_count_categories: Returns series of all images that are considered annotated using annotations_df, RequiredAnnotations, 
    and RequiredAmounts 
    
categorise_annotations: returns 4 dataframes contaning image ID and groundtruth: combinations of: unannotated, annotated, benign and malignant. 
    This is done using a series of images id's that are considered annotated and paths to the groundtruths.  

create_group_sets: Returns a dict containing dataframes for each group with assigned ISIC ID's and groundtruths. This is based on the 4 
    dataframes explained in the previous function and NumGroups NumImages AnnotationBalance.
    
save_group_sets: saves image id's (with and without ground truth) as csv to the folder SavePath, using the dict containing dataframes
    with image id's and groundtruth.
