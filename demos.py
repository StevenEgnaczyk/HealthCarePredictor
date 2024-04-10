import numpy
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
from datetime import datetime 
from sklearn.preprocessing import OneHotEncoder


def main():
    # Grab the data from the .csv files
    demos_data = pd.read_csv("train/train_demos.csv")
    labels = pd.read_csv("train/train_labels.csv")
    # DEBUG: let's see what the data looks like
    print(demos_data.head())
    print()
    print(labels.head())
    # END DEBUG

    # Combine demos_data and labels by their patient_id's
    merged_data = pd.merge(demos_data, labels, on="patient_id")
    # DEBUG: print the merged data
    print()
    print(merged_data.head())
    # END DEBUG

    # Parse out time from admittime -- use datetime package
    # Using only the hour value (the date, minutes and seconds not as important)
    for full_time in merged_data['admittime']:
        datetime_object = datetime.strptime(full_time, '%Y-%m-%d %H:%M:%S')
        merged_data['admittime'] = merged_data['admittime'].replace([full_time], datetime_object.time().hour)
    # DEBUG: print updated times
    print()
    print(merged_data.head())
    # END DEBUG

    # Deal with the categorical data (gender, marital status, race, insurance)
    encoder = OneHotEncoder()
    encoded_array = encoder.fit_transform(merged_data[['gender', 'insurance', 'marital_status', 'ethnicity']]).toarray()
    encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['gender', 'insurance', 'marital_status', 'ethnicity']))
    final_data = pd.concat([encoded_data, merged_data.drop(columns=['gender', 'insurance', 'marital_status', 'ethnicity'])], axis=1)
    del final_data['patient_id']
    # DEBUG: print newly encoded data set
    print()
    print(final_data.head())
    # END DEBUG


    # Get Y (X = final_data, Y = labels array)
    Y = final_data["label"].to_numpy()


    # Split the dataset (training and validation)
    # Dataset Splitting:
    X_train, X_test, Y_train, Y_test = train_test_split(final_data, Y, train_size=0.8, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=0)

    # Run a classifier (Decision Tree) on the dataset
    dt_clf = DecisionTreeClassifier(random_state=0)
    dt_clf.fit(X_train, Y_train)

    # Calculate important values
    # For the Test Set:
    test_acc = dt_clf.score(X_test, Y_test)
    y_pred = dt_clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    spec = tn / (tn + fp)
    auroc = roc_auc_score(Y_test, dt_clf.predict_proba(X_test)[:,1])
    print("Test Values: ")
    print("Accuracy: ", test_acc)
    print("Precision: ", prec)
    print("Recall: ", recall)
    print("Specificity: ", spec)
    print("AUROC: ", auroc)
    print()

    # For the Validation Set:
    val_acc = dt_clf.score(X_val, Y_val)
    y_pred = dt_clf.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(Y_val, y_pred).ravel()
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    spec = tn / (tn + fp)
    auroc = roc_auc_score(Y_val, dt_clf.predict_proba(X_val)[:,1])
    print("Validation Values: ")
    print("Accuracy: ", val_acc)
    print("Precision: ", prec)
    print("Recall: ", recall)
    print("Specificity: ", spec)
    print("AUROC: ", auroc)


main()