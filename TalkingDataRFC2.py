import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
#==============================================================================================================================================
print("# Read App Labels")
app_lab = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/app_labels.csv", dtype={'device_id': np.str})
app_lab = app_lab.groupby("app_id")["label_id"].apply(
    lambda x: " ".join(str(s) for s in x))
#==============================================================================================================================================
print("Joining the pieces of app_events.csv into one app_events.csv file....")

app1 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/app_events.zip.001",sep='delimiter', engine='python',header=None)
app2 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/app_events.zip.002",sep='delimiter', engine='python',header=None)
app3 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/app_events.zip.003",sep='delimiter', engine='python',header=None)
app4 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/app_events.zip.004",sep='delimiter', engine='python',header=None)
app5 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/app_events.zip.005",sep='delimiter', engine='python',header=None)

d1 =pd.DataFrame(app1)
d2 =pd.DataFrame(app2)
d3 =pd.DataFrame(app3)
d4 =pd.DataFrame(app4)
d5 =pd.DataFrame(app5)

app_frames = [d1, d2, d3, d4, d5]
app_ev = pd.concat(app_frames)

app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
app_ev = app_ev.groupby("event_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_lab
#==============================================================================================================================================
print("Joining the pieces of events.csv into one events.csv file....")

eve1 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/events.csv.001",dtype={'device_id': np.str})
eve2 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/events.csv.002",dtype={'device_id': np.str})
eve3 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/events.csv.003",dtype={'device_id': np.str})
eve4 = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/events.csv.004",dtype={'device_id': np.str})

df1 =pd.DataFrame(eve1)
df2 =pd.DataFrame(eve2)
df3 =pd.DataFrame(eve3)
df4 =pd.DataFrame(eve4)
frames = [df1, df2, df3, df4]
events = pd.concat(frames)

events["app_lab"] = events["event_id"].map(app_ev)
events = events.groupby("device_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_ev
#================================================================================================================================================
print("# Read Phone Brand")

pbd = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)
#================================================================================================================================================
print("# Generate Train and Test")

train = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/gender_age_train.csv",
                    dtype={'device_id': np.str})
train["app_lab"] = train["device_id"].map(events)

train = pd.merge(train, pbd, how='left',
                 on='device_id', left_index=True)

test = pd.read_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/gender_age_test.csv",
                   dtype={'device_id': np.str})
test["app_lab"] = test["device_id"].map(events)

test = pd.merge(test, pbd, how='left',
                on='device_id', left_index=True)

del pbd
del events
#================================================================================================================================================
def get_hash_data(df):
     hasher = FeatureHasher(input_type='string')
     # hasher = DictVectorizer(sparse=False)
     df = df[["phone_brand", "device_model", "app_id"]].apply(
         lambda x: ",".join(str(s) for s in x), axis=1)
     df = hasher.transform(df.apply(lambda x: x.split(",")))
     return df

df = df[["phone_brand", "device_model", "app_lab"]].astype(np.str).apply(
        lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
df_tfv = tfv.fit_transform(df)

train = df_tfv[:split_len, :]
test = df_tfv[split_len:, :]
return train, test

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)

device_id = test["device_id"].values
train, test = get_hash_data(train,test)

X_train, X_val, y_train, y_val = train_test_split(train, Y, train_size=.80)

#==================================================================================================================================================

    
