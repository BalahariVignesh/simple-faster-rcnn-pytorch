from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def auroc(preds, labels):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def aupr(preds, labels):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    precision, recall, _ = precision_recall_curve(labels, preds)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    # Get the index of the first TPR that is >= 95%
    idx = next(i for i, x in enumerate(tpr) if x>=0.95)
    
    return fpr[idx]


def detection_error(preds, labels):
    """Return the misclassification probability when TPR is 95%.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    # Get the index of the first TPR that is >= 95%
    idx = next(i for i, x in enumerate(tpr) if x>=0.95)
    
    t = tpr[idx]
    f = fpr[idx]
    
    return 0.5 * (1 - t) + 0.5 * f


def plot_roc(preds, labels, title="Receiver operating characteristic"):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """
    
    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)
    
    # Compute FPR (95% TPR)
    idx = next(i for i, x in enumerate(tpr) if x>=0.95)
    fpr_at_95_tpr = fpr[idx]
    
    # Compute AUROC
    roc_auc = auc(fpr, tpr)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % fpr_at_95_tpr)
    plt.plot([fpr_at_95_tpr, fpr_at_95_tpr], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    
def plot_prc(preds, labels, title="Precision recall curve"):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """
    
    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
#     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    

def get_summary_statistics(predictions, labels):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.
    
    These metrics conform to how results are reported in the paper 'Enhancing The 
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.
    
        preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    
    return {
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels),
        'detection_error': detection_error(predictions, labels),
        'auroc': auroc(predictions, labels),
        'aupr_in': aupr(predictions, labels),
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels])
    }


def html_summary_table(data):
    """Generate HTML table of novelty detection statistics.
    
    data: dict
    A JSON like structure of the table data that has the following format:
    
    data = {
        'modelA': {
            'inlier_name': 'MNIST'
            'outliers': {
                'Fashion MNIST': {
                    'fpr_at_95_tpr': 0.02123,
                    'detection_error': 0.02373,
                    'auroc': 0.96573,
                    'aupr_in': 0.91231,
                    'aupr_out': 0.9852
                },
                'EMNIST Letters': {
                    'fpr_at_95_tpr': 0.02123,
                    'detection_error': 0.02373,
                    'auroc': 0.96573,
                    'aupr_in': 0.91231,
                    'aupr_out': 0.9852,
                }
            }
        },
        
        'modelB': {
            'inlier_name': 'MNIST'
            'outliers': {...}            
        }      
    }
    """
    
    table = """
        <table>
            <tr>
                <th>Model</th>
                <th>Out-of-distribution dataset</th>
                <th>FPR (95% TPR)</th>
                <th>Detection Error</th>
                <th>AUROC</th>
                <th>AUPR In</th>
                <th>AUPR Out</th>
            </tr>
    """

    for i, (model, model_data) in enumerate(data.items()):
        table += "<tr>"
        table += "<td rowspan={}><b>{}</b> ({})</td>".format(len(model_data['outliers']), model, model_data['inlier_name'])    

        for j, (outlier_name, scores) in enumerate(model_data['outliers'].items()):
            if j != 0:
                table += "<tr>"

            table += "<td>{}</td>".format(outlier_name)
            table += "<td>{:.1f}</td>".format(scores['fpr_at_95_tpr'] * 100)
            table += "<td>{:.1f}</td>".format(scores['detection_error'] * 100)
            table += "<td>{:.1f}</td>".format(scores['auroc'] * 100)
            table += "<td>{:.1f}</td>".format(scores['aupr_in'] * 100)
            table += "<td>{:.1f}</td>".format(scores['aupr_out'] * 100)
            table += "</tr>"

    table += "</table>"
    
    return table


def barcode_plot(preds, labels):
    """Plot a visualization showing inliers and outliers sorted by their prediction of novelty."""
    # the bar
    x = sorted([a for a in zip(preds, labels)], key=lambda x: x[0])
    x = np.array([[49,163,84] if a[1] == 1 else [173,221,142] for a in x])
    # x = np.array([a[1] for a in x]) # for bw image
    
    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary_r, interpolation='nearest')

    fig = plt.figure()

    # a horizontal barcode
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x.reshape((1, -1, 3)), **barprops)

    plt.show()