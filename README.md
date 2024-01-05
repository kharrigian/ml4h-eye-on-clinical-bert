# An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping

This is the official code repository for the ML4H (Findings) paper ["An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping."](https://arxiv.org/abs/2311.08687v1) If you use any of the code in this repository, we kindly ask that you cite the following reference:

```bibtex
@article{harrigian2023eye,
  title={An Eye on Clinical BERT: Investigating Language Model Generalization for Diabetic Eye Disease Phenotyping},
  author={Harrigian, Keith and Tang, Tina and Gonzales, Anthony and Cai, Cindy X and Dredze, Mark},
  journal={arXiv preprint arXiv:2311.08687},
  year={2023}
}
```

If you have questions or issues when working with this repository, please submit a GitHub issue or [email us directly](mailto:kharrigian@jhu.edu).

## Abstract

Diabetic eye disease is a major cause of blindness worldwide. The ability to monitor relevant clinical trajectories and detect lapses in care is critical to managing the disease and preventing blindness. Alas, much of the information necessary to support these goals is found only in the free text of the electronic medical record. To fill this information gap, we introduce a system for extracting evidence from clinical text of 19 clinical concepts related to diabetic eye disease and inferring relevant attributes for each. In developing this ophthalmology phenotyping system, we are also afforded a unique opportunity to evaluate the effectiveness of clinical language models at adapting to new clinical domains. Across multiple training paradigms, we find that BERT language models pretrained on out-of-distribution clinical data offer no significant improvement over BERT language models pretrained on non-clinical data for our domain. Our study tempers recent claims that language models pretrained on clinical data are necessary for clinical NLP tasks and highlights the importance of not treating clinical language data as a single homogeneous domain.

## Repository Overview

This code repository has been adapted from an internal research code repository. It includes core functionality examined within our paper, including clinical concept extraction and attribute classification. We have not included code which was used to conduct all of our experiments (e.g., hyperparameter sweeps, plotting), as much of it is not generalizable beyond our insitution's compute server. In the process of consolidating code from the internal research repository, it is possible that certain bugs were introduced. We recommend using our code repository as a starting point and encourage deep inspection.

## Data and Models

At this time, data and models from our study are not able to be released due to patient privacy constraints. If this changes in the future, we will provide additional information here.

In the interim, we have gathered two synthetic datasets which can be used for testing functionality of the code base.

* `data/resources/synthetic-notes/`: We used GPT Turbo 3.5 to generate synthetic encounter notes that have the same format as notes in our internal dataset. These notes have not been evaluated for medical accuracy and are intended only to showcase the creation of our annotation worksheets.
* `data/resources/synthetic-wnut_2017`: We transformed the W-NUT 2017 Shared Task dataset into the format expected for most functionality in our code base (token-span classification).

Users of this code repository may refer to these synthetic datasets as a guide for formatting their own clinical note datasets. As shown by the usage of the W-NUT Shared Task dataset, our code can also be adapted to support non-clinical applications with relative ease.

## Installation

To use code in this repository, you should start by installing necessary dependencies and the `cce` library. The easiest way to do this is by invoking the following command from the root of this repository.

```bash
pip install -e .
```

We recommend using `conda` to manage your environment. We developed all code using Python 3.10 on a unix-based remote server. GPU access is highly recommended for language model and classifier training, but not required.

## Usage

Each subsection below details a functionality provided by this code repository. We have compiled several bash files in the `utilities/` directory which include example system commands for each functionality. The series of commands presented below is intended to present the expected runtime order, not necessarily to be invoked verbatim.

### Clinical Concept Extraction

To see how to extract clinical concepts from free text, please see this [notebook](./notebooks/concept_extraction.ipynb).

### Clinical Concept Annotation

We have included code to facilitate clinical concept annotation within a human-readable Excel workbook. Once an annotator has assigned labels within the notebook, they can generate a formatted JSON file which can be used for model training.

```bash
## Prepare Synthetic Note Data
./utilities/prepare_synthetic_notes.sh

## Extract Concepts from Synthetic Notes and Generate a Worksheet for Annotation
./utilities/prepare_worksheet.sh

## Transform the Annotated Excel worksheet into JSON format for model training
./utilities/transform_worksheet.sh
```

### BERT Pretraining

One of the core findings from our study is that domain adaptation is necessary for maximizing downstream task performance when using existing language models. We provide boilerplate code to facilitate pretraining of a BERT language model. Users can swap out the synthetic dataset used for testing with their own by changing appropriate filepaths in the CLI command.

```bash
## Download W-NUT 2017 Data and Transform for Model Training
./utilities/prepare_synthetic_wnut.sh

## Train a BERT Model From Scratch with a Domain-Specific Tokenizer
./utilities/pretrain_scratch_tokenizer_and_model.sh

## Train a BERT Model From Scratch with an Existing Tokenizer
./utilities/pretrain_scratch_model.sh

## Continue Pretraining an existing BERT Model
./utilities/pretrain_continue.sh
```

### Named Entity Recognition and Attribute Classification

Our phenotyping system relies on two stages: 1) concept extraction and 2) attribute classification. We found regular expressions to be sufficient for the first stage, but provide code to train an NER model from scratch as well. In the context of our study, we train attribute classifiers to infer laterality, severity, type, and status associated with the extracted clinical concepts.

```bash
## Download W-NUT 2017 Data and Transform for Model Training
./utilities/prepare_synthetic_wnut.sh

## Train a Named Enitity Recognition Model on W-NUT Data
./utilities/train_model_entity.sh

## Train an Attribute Classification Model on W-NUT Data
./utilities/train_model_attributes.sh

## Generate Performance Visualizations for W-NUT Models
./utilities/train_aggregate.sh
```