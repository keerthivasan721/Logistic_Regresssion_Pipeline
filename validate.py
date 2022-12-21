from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score


def validate_model(ytrue,ypreds):
    print("Precision",precision_score(ytrue,ypreds,average="micro"))
    print("Recall",recall_score(ytrue,ypreds,average="micro"))
    print("Accuracy",accuracy_score(ytrue,ypreds))
    print("Confusion matrix:","\n",confusion_matrix(ytrue,ypreds))
