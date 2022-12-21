import preprocess as prep
import model
import pandas as pd
import prediction
import validate

def load_dataset(x):
    return  pd.read_csv(x)


df=load_dataset(r"dataset\fish_train.csv")
df.columns=[i.lower() for i in df.columns]
x,y=prep.split(df)
xtrain,xtest,ytrain,ytest=prep.train_test(x,y)



# lable encoding

ytrain=prep.l_encode(ytrain,is_train=True)
ytest=prep.l_encode(ytest,is_train=False)


# Multicollinearity Test

xtrain,vif=prep.check_multicollinearity(xtrain)

# Since few columns are removed from train data due to multicollinearity, let us remove those 
# columns from test data as well

xtest=xtest[xtrain.columns]




#  scaling the independent variable

xtrain=prep.scaler(xtrain,is_train=True)
xtest=prep.scaler(xtest,is_train=False)



#  Model Buiding using sklearn.logistic regression

sk_model= model.sk_logistic(xtrain,ytrain)

# Making predictions

ypreds=prediction.make_prediction(sk_model,xtest)

print("Predicted values",ypreds)

# validating Model

validate.validate_model(ytest,ypreds)


