train_demos:
load in patient information
Parse out time from admittime
One-hot encode -> insurance, marital_status, ethnicity
Deal with non-values in marital_status

test_signs -
Load in information
Take the average, min, and max of the features
Do some sort of dimensionality reduction or feature selection to narrow down the features
Linked by the patient_id

test_radiology:
Find some way to parse through all of the radiology reports and tokenize them. and then hack together code from assignment 1 to determine sentiment, should be a value 0 to 1.
