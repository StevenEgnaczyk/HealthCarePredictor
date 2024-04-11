import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay

# Calculate and display the performance metrics for the given classifier using the given testing set
def testClassifier(classifier, testing, scores, plot_title):
    # Predict the scores for the given testing set and record the time it takes
    start_time = time.time()
    predictions = classifier.predict(testing)
    end_time = time.time()
    online_efficiency_cost = end_time - start_time
    # Calculate performance metrics
    accuracy = accuracy_score(scores, predictions)
    precision = precision_score(scores, predictions)
    recall = recall_score(scores, predictions)
    specificity = recall_score(scores, predictions, pos_label=0)
    auroc = roc_auc_score(scores, predictions)
    # Display performance metrics and curves
    print("Online Efficiency Cost:", online_efficiency_cost)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("AUROC:", auroc)
    plotPerformanceCurves(scores, predictions, plot_title)

# Plot the Feature Importance graph for the given classifier and feature names
def plotFeatureImportances(classifier, feature_names):
    importances = classifier.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

# Plot the ROC Curve and Precision-Recall Curve for the given classifier results
def plotPerformanceCurves(scores, predictions, plot_title):
    roc = RocCurveDisplay.from_predictions(scores, predictions)
    precision_recall = PrecisionRecallDisplay.from_predictions(scores, predictions)
    roc.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve: {plot_title}",
    )
    precision_recall.ax_.set(
        xlabel="Precision",
        ylabel="Recall",
        title=f"Precision-Recall Curve: {plot_title}",
    )
    plt.tight_layout()

if __name__ == '__main__':
    # Load training data and labels
    signs_datafile = pd.read_csv("train/train_signs.csv")
    labels_datafile = pd.read_csv("train/train_labels.csv")

    # Calculate mean, min, and max for each vital sign for each patient
    num_cols = signs_datafile.select_dtypes(include=[np.number]).columns
    stats = signs_datafile.groupby("patient_id")[num_cols].agg(['mean','max','min'])
    stats.columns = ['_'.join(col) for col in stats.columns]

    # Get label and features for each patient
    labels = labels_datafile['label'].values
    features = stats.values
    feature_names = [f"{sign}" for sign in stats.columns]

    # Split data into training and validation sets
    train, validation, train_labels, validation_labels = train_test_split(features, labels, test_size=0.2, train_size=0.8)

    # Train classifier on the data
    classifier = DecisionTreeClassifier(random_state=0)
    start_time = time.time()
    classifier.fit(train, train_labels)
    end_time = time.time()
    offline_efficiency_cost = end_time - start_time

    # Calculate and plot the importance of each feature
    plotFeatureImportances(classifier, feature_names)

    # Run classifier on the training and validation data
    print("\nOffline Efficiency Cost: ", offline_efficiency_cost)
    print("\n-----------Training Set Metrics-----------")
    testClassifier(classifier, train, train_labels, "Decision Tree on Training Data")
    print("\n----------Validation Set Metrics----------")
    testClassifier(classifier, validation, validation_labels, "Decision Tree on Validation Data")

    # Load in test data and calculate mean, max, and main
    test_datafile = pd.read_csv("test/test_signs.csv")
    num_cols = test_datafile.select_dtypes(include=[np.number]).columns
    test_data = test_datafile.groupby("patient_id")[num_cols].agg(['mean','max','min'])
    test_data.columns = ['_'.join(col) for col in test_data.columns]

    # Run classifier on test data
    test = test_data.values
    testing_predictions = classifier.predict_proba(test)

    # Create DataFrame for each patient and their predicted probabilities
    test_data.reset_index(inplace=True)
    results = pd.DataFrame(testing_predictions, columns=classifier.classes_)
    results["patient_id"] = test_data.index
    results = results[["patient_id"] + list(classifier.classes_)]

    # Display all the figures created
    plt.show()

    # Save results to a CSV file
    results.to_csv("signs_predictions.csv", index=False)
