# An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping

This is the official code repository for the ML4H (Findings) paper "An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping." If you use any of the code in this repository, we kindly ask that you cite the following reference:

```
@article{harrigian2023eye,
  title={An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping},
  author={Harrigian, Keith and Tang, Tina and Gonzales, Anthony and Cai, Cindy X and Dredze, Mark},
  journal={arXiv preprint arXiv:2311.08687},
  year={2023}
}
```

If you have questions or issues when working with this repository, please submit a GitHub issue or [email us directly](mailto:kharrigian@jhu.edu).

## Overview




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