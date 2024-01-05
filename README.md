# An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping

This is currently a placeholder for what will become the home of code (and maybe data/models) from our ML4H (Findings) paper "An Eye on 
Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping."

If you would like to be notified when we update this repository with relevant resources, please either star/watch the repo using the GitHub 
UI, or [email us directly](mailto:kharrigian@jhu.edu) to indicate your interest.


## Pipeline

```
## Format the synthetic note data
./utilities/prepare_synthetic_notes.sh
## Create an Excel worksheet to use for annotation
./utilities/prepare_worksheet.sh
## Transform annotated Excel worksheet into JSON format for model training
./utilities/transform_worksheet.sh
```

```
## Download W-NUT 2017 Data and Transform for Model Training
./utilities/prepare_synthetic_wnut.sh
## Train an Enitity Recognition Model on W-NUT Data
./utilities/train_model_entity.sh
## Train an Attribute Classification Model on W-NUT Data
./utilities/train_model_attributes.sh
## Generate Performance Visualizations for W-NUT Models
./utilities/train_aggregate.sh
```