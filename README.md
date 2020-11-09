This code processes annotations for ISIC 2017 lesion images from previous years, and assigns unannotated images to the new groups.
annotation_processing_script.py calls all the functions that are neccesary to assign the images to the new groups. The following
variables need to be set by the user:

# Requirements

pandas>=1.0.5
numpy>=1.18.5

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

The following data/annotation types can be chosen in RequiredAnnotations. Only Asymmetry, Border and Color contain a significant amount of annotations (around 10000 each). 
The output of the function annotation_stats is shown first:

|data_type|ann_count                    |mean_ann_img|min_ann_img                                  |
|---------|-----------------------------|------------|---------------------------------------------|
|Asymmetry|10471                        |6.4042813455657495|3                                            |
|Black    |1099                         |5.495       |2                                            |
|Blood    |250                          |2.5         |2                                            |
|Blue     |1197                         |3.99        |3                                            |
|Border   |10762                        |6.582262996941896|3                                            |
|Brown    |800                          |8.0         |8                                            |
|Color    |10175                        |6.223241590214068|3                                            |
|Color_Categorised|1599                         |3.4166666666666665|3                                            |
|Color_Fade|1000                         |5.0         |3                                            |
|Compactness|1764                         |5.297297297297297|3                                            |
|Dermo    |1485                         |3.75        |3                                            |
|Diameter |295                          |2.9797979797979797|1                                            |
|Erythema |664                          |4.992481203007519|4                                            |
|Flaking  |300                          |3.0         |3                                            |
|Red      |1100                         |5.5         |3                                            |
|Rough_Surface|300                          |3.0         |3                                            |
|Skin_Color|299                          |2.99        |2                                            |
|White    |300                          |3.0         |3                                            |
|Yellow   |800                          |8.0         |8                                            |


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
