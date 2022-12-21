from sklearn.linear_model import LogisticRegression
from statsmodels.api import Logit
import pickle


def sk_logistic(x,y):
    log_model=LogisticRegression().fit(x,y)
    with open("artifacts//%s.pickle" %log_model,"wb") as m:
        pickle.dump(log_model,m)
    return log_model
        