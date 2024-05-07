DATA
=========
Train and test dataset is a table in CSV file. Columns are features + label. </br>
For example:
| balance | n_total | n_uniq |  label  |
|-----|-----|-----|---------|
|798 | 893 | 321 | mixer |
|320 | 776 | 189 | gambling |

> See another example in `data/processed/train.v1.csv`

#### RULE 1
Train and test feature values must be **already preprocessed**. They go to ML model without transformation. Preprocess is used for inference only. </br>

#### RULE 2
All labels in CSV must be from list in `config:classes` </br>

#### Recomendation
Each CSV file should have a **version** reference, like train.v1.csv.
