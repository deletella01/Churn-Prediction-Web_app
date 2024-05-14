import streamlit as st
import pandas as pd
import pickle

st.write("""
# Churn Prediction and Customer Retention App
This app forecasts consumer behavior. The objective is to create a targeted client retention program by analyzing all pertinent customer data.

Data obtained from [Kaggle Library](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
""")

st.sidebar.header("User Input Features")

def user_input_Features():
  gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
  seniorCitizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
  partner = st.sidebar.selectbox('Partner',('Yes', 'No'))
  dependents = st.sidebar.selectbox('Dependents',('Yes', 'No'))
  tenure = st.sidebar.slider('Tenure(Years))', 0, 75, 20)
  PhoneService = st.sidebar.selectbox('Phone Service',('Yes', 'No'))
  MultipleLines = st.sidebar.selectbox('Multiple Lines',('Yes', 'No'))
  OnlineSecurity = st.sidebar.selectbox('Online Security',('Yes', 'No'))
  OnlineBackup = st.sidebar.selectbox('Online Backup',('Yes', 'No'))
  DeviceProtection = st.sidebar.selectbox('Device Protection',('Yes', 'No'))
  TechSupport = st.sidebar.selectbox('Tech Support',('Yes', 'No'))
  StreamingTV = st.sidebar.selectbox('Streaming TV',('Yes', 'No'))
  StreamingMovies = st.sidebar.selectbox('Streaming Movies',('Yes', 'No'))
  PaperlessBilling = st.sidebar.selectbox('Paperless Billing',('Yes', 'No'))
  MonthlyCharges = st.sidebar.slider('Monthly Charge', 18, 118, 70)
  TotalCharges = st.sidebar.slider('Total Charge', 18, 8685, 1000)

  InternetService = st.sidebar.selectbox('Internet Service',('DSL','Fiber Optic','No'))
  Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
  PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

  data = {'gender' : gender,
          'seniorCitizen' : seniorCitizen,
          'Partner' : partner,
          'Dependents' : dependents,
          'tenure' : pd.to_numeric(tenure),
          'PhoneService' : PhoneService,
          'MultipleLines' : MultipleLines,
          'OnlineSecurity' : OnlineSecurity,
          'OnlineBackup' : OnlineBackup,
          'DeviceProtection' : DeviceProtection,
          'TechSupport' : TechSupport,
          'StreamingTV' : StreamingTV,
          'StreamingMovies' : StreamingMovies,
          'PaperlessBilling' : PaperlessBilling,
          'MonthlyCharges' : pd.to_numeric(MonthlyCharges),
          'TotalCharges' : pd.to_numeric(TotalCharges),
          'InternetService' : InternetService,
          'Contract' : Contract,
          'PaymentMethod' : PaymentMethod}
  features = pd.DataFrame(data, index=[0])
  return features

d = user_input_Features()

st.subheader('User Input Parameters')
st.write(d)


def data_trans4m(df):
  yes_no_columns = ['Partner', 'seniorCitizen','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
  for col in yes_no_columns:
    df[col].replace({'Yes': 1,'No': 0},inplace=True)

  df['gender'].replace({'Female':1,'Male':0},inplace=True)

  df2 = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod'])

  bool_cols = df2.select_dtypes(include=[bool]).columns
  df2[bool_cols] = df2[bool_cols].astype('uint8')

  cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
  return df2

d2 = data_trans4m(d)

if 'InternetService_DSL' in d2.columns:
  d2['InternetService_Fiber optic'] = 0
  d2['InternetService_No'] = 0
elif 'InternetService_Fiber_optic' in d2.columns:
  d2['InternetService_DSL'] = 0
  d2['InternetService_No'] = 0
else:
  d2['InternetService_DSL'] = 0
  d2['InternetService_Fiber optic'] = 0

if 'Contract_Month-to-month' in d2.columns:
  d2['Contract_One year'] = 0
  d2['Contract_Two year'] = 0
elif 'Contract_One year' in d2.columns:
  d2['Contract_Month-to-month'] = 0
  d2['Contract_Two year'] = 0
else:
  d2['Contract_Month-to-month'] = 0
  d2['Contract_One year'] = 0

if 'PaymentMethod_Bank transfer (automatic)' in d2.columns:
  d2['PaymentMethod_Credit card (automatic)'] = 0
  d2['PaymentMethod_Electronic check'] = 0
  d2['PaymentMethod_Mailed check'] = 0
elif 'PaymentMethod_Credit card (automatic)' in d2.columns:
  d2['PaymentMethod_Bank transfer (automatic)'] = 0
  d2['PaymentMethod_Electronic check'] = 0
  d2['PaymentMethod_Mailed check'] = 0
elif 'PaymentMethod_Electronic check' in d2.columns:
  d2['PaymentMethod_Bank transfer (automatic)'] = 0
  d2['PaymentMethod_Credit card (automatic)'] = 0
  d2['PaymentMethod_Mailed check'] = 0
else:
  d2['PaymentMethod_Bank transfer (automatic)'] = 0
  d2['PaymentMethod_Credit card (automatic)'] = 0
  d2['PaymentMethod_Electronic check'] = 0

# Reads in saved classification model
load_model = pickle.load(open('churn_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(d2)


st.subheader('Prediction')
if prediction > 0.5:
  st.write('YES')
  st.write('Customer will leave the company')
else:
  st.write('NO')
  st.write('Customer will stay with the company')
