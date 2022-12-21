import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def split(df):
    x=df.iloc[:,1:]
    y=df.iloc[:,[0]]
    return x,y

def train_test(x,y):
    if y.value_counts(normalize=True).values.min()<0.25:
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,stratify=y,random_state=1)
        return xtrain,xtest,ytrain,ytest
    else:
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
        return xtrain,xtest,ytrain,ytest




def l_encode(df,is_train=True):
    if is_train==True:
        if df.dtypes[0]==object:
            label_encoder= LabelEncoder().fit(df)
            with open("artifacts//label_encoder.pickle","wb") as w:
                pickle.dump(label_encoder,w)
            return pd.DataFrame(label_encoder.transform(df),columns=list(df.columns))

    else:
         if df.dtypes[0]==object:
            with open("artifacts\label_encoder.pickle","rb") as w:
                label_encod=pickle.load(w)
            return pd.DataFrame(label_encod.transform(df),columns=list(df.columns))


def scaler(df,is_train=True):
    if is_train==True:
        if object not in df.dtypes.values :
            minmax_obj=MinMaxScaler().fit(df)
            with open("artifacts//minmax_scaler.pickle","wb") as d:
                pickle.dump(minmax_obj,d)
            return pd.DataFrame(minmax_obj.transform(df),columns=df.columns)
    else:
        if object not in df.dtypes.values:
            with open("artifacts//minmax_scaler.pickle","rb") as d:
                minmaxobj=pickle.load(d)
            return pd.DataFrame(minmaxobj.transform(df),columns=df.columns)

def check_multicollinearity(df):
    while True:
        final_df=pd.DataFrame(columns=["feature","vif"])
        for i,j in zip(df.columns,range(len(df.columns))):
            # final_df=final_df.append({ "feature":i,"vif":variance_inflation_factor(df,j)},ignore_index=True)
            final_df=pd.concat((final_df, pd.DataFrame({"feature":[i],"vif":[variance_inflation_factor(df,j)]})),axis=0)

        if final_df["vif"].min()>5:
            rr=final_df.loc[final_df["vif"]==final_df["vif"].max(),"feature"]
            df=df.drop(rr.values[0],axis=1)
        else:
            return df,final_df


