import json
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, accuracy_score


def analyze_list(predictions):
    print("analyzing the list values in predictions")
    true_values = []
    pred_values = []
    prob_values = []
    for prediction in predictions:
        true_values.append(prediction[0])
        pred_values.append(prediction[1])
        if len(prediction)>2:
            prob_values.append(prediction[2])
    
    return true_values, pred_values, prob_values


def get_predictions_vals(report_data):
    """
    Extract the prediction values from the reports.

    Returns: true values, preducted values, probability values
    """
    cross_val = report_data.get("cross-validation", False)
    true_values = []
    pred_values = []
    prob_values = []
    if cross_val:        
        predictions = report_data.get("predictions")
        
        for iter in predictions:
                true, pred, prob = analyze_list(iter)
                true_values.extend(true)
                pred_values.extend(pred)
                prob_values.extend(prob)
    else:
        predictions = report_data.get("predictions")
        true_values, pred_values, prob_values = analyze_list(predictions)

    return true_values, pred_values, prob_values


def device_wise_confusion_matrix(device_wise_cm, plots_path):
    """
    Generate device-wise confusion matix plots.
    """
    
    for device in device_wise_cm:
        display = ConfusionMatrixDisplay(device_wise_cm[device])
        display.plot(values_format='.1f')
        plt.savefig(os.path.join(plots_path, device + "-CM.png"), dpi=300, bbox_inches = "tight")
        plt.savefig(os.path.join(plots_path, device + "-CM.pdf"), dpi=300, bbox_inches = "tight")
        plt.close()


def extract_stats(report_data, report_path):
    """
    Extract statistics from the reports and generate confusion matrix

    Returns: overall accuracy and device_wise confusion matrix
    """
    true_values, pred_values, prob_values = get_predictions_vals(report_data)
    print(len(true_values), len(pred_values))

    if type(pred_values[0]) == list:
        is_multilabel = True
    else:
        is_multilabel = False

    if is_multilabel:
        label_values = sorted(list(set(x[0] for x in true_values)))

        mlb = MultiLabelBinarizer()
        true_values_mlb = mlb.fit_transform(true_values)
        pred_values_mlb = mlb.transform(pred_values)
        print(true_values_mlb.shape)
        print(pred_values_mlb.shape)
        
        accuracy = accuracy_score(true_values_mlb, pred_values_mlb)
        print("Accuracy score for the evaluation: ", accuracy)
        # Get the confusion matrix using the values
        confusion_matrix = multilabel_confusion_matrix(true_values_mlb, pred_values_mlb)

    else:
        # Get the list of labels    
        label_values = list(sorted(set(true_values)))
        
        accuracy = accuracy_score(true_values, pred_values)
        print("Accuracy score for the evaluation: ", accuracy)
        # Get the confusion matrix using the values
        confusion_matrix = multilabel_confusion_matrix(true_values, pred_values, labels=label_values)
        # Generate a plot of the confusion matrix
        matrix = ConfusionMatrixDisplay.from_predictions(true_values,
                                                     pred_values,
                                                     labels=label_values,
                                                     normalize='true')
        fig, ax = plt.subplots(figsize=(25,25))
        matrix.plot(ax=ax, xticks_rotation="vertical", values_format='.2f')
   
    
    report_data["accuracy"] = accuracy
    fh = open(report_path, "w")
    fh.write(json.dumps(report_data, indent=2))

    # Store the confusion matrix in the same folder as the report file
    report_num = os.path.basename(report_path).split(".")[0].split("_")[-1]
    if "training-data" in report_data:
        plots_dir = report_data["training-data"] + "_" + report_data["testing-data"] + "_" + report_num
    else:
        plots_dir = report_data["eval-config"]["train-dataset"] + "_" + report_data["eval-config"]["test-dataset"] + "_" + report_num
    
    plots_path = os.path.join(os.path.dirname(report_path), plots_dir)

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    device_wise_cm = {}
    for idx, label in enumerate(label_values):
        device_wise_cm[label] = confusion_matrix[idx]
    
    device_wise_confusion_matrix(device_wise_cm, plots_path)

    print(confusion_matrix)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(plots_path, "multi-confusion-matrix.png"), bbox_inches = "tight")
    plt.savefig(os.path.join(plots_path, "multi-confusion-matrix.pdf"), bbox_inches = "tight")
    plt.legend()
    plt.close()
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR! Require path to the report file as argument!")
        print("Usage: python analyze-report.py <path-to-report-file>")
        exit(1)

    report_path = sys.argv[1]
    # Check if path exists for report 1
    if not os.path.exists(report_path):
        print("ERROR! Path to the Report does NOT exist!")
        exit(1)
    if not report_path.endswith(".json"):
        print("ERROR! Wrong file path entered for Report! Require a JSON file type.")
        exit(1)

    report_data = json.load(open(report_path, "r"))

    print(report_data)

    extract_stats(report_data, report_path) 
