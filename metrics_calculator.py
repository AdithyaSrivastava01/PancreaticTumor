from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

def calculate_metrics(model, X_test, y_test, threshold=0.5):
    """
    Calculate various classification metrics for a given model.

    Parameters:
    - model: The trained classification model.
    - X_test: The feature vectors of the test set.
    - y_test: The true labels of the test set.
    - threshold: The decision threshold for binary classification (default is 0.5).

    Returns:
    - metrics_dict: A dictionary containing various classification metrics.
    """
    # Make predictions on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, y_pred)

    # Calculate sensitivity (recall) and specificity
    tp, fn, fp, tn = confusion_mat.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics_dict = {
        'Accuracy': accuracy,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'Confusion Matrix': confusion_mat,
        'Classification Report': classification_rep,
        'ROC AUC': roc_auc,
        'F1 Score': f1
    }

    return metrics_dict
