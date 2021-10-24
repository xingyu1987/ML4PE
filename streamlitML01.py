import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] =False
from joblib import dump, load
import os
import shap
st.set_page_config(
     page_title="Machine learning for pulmonary embolism",
     page_icon="random",
     layout="centered",
     initial_sidebar_state="expanded",

 )



with st.sidebar.form("Features Input"):
    st.subheader("Features Input")
    Platelet  = st.slider("Platelet",min_value=0,max_value=400, value=100)
    Albumin   = st.slider("Albumin",min_value=20,max_value=50,value=25)
    Cholesterol= st.slider("Cholesterol",min_value=1,max_value=10,value=2)
    Sodium = st.slider("Sodium",min_value=120,max_value=170,value=135)
    APTT = st.slider("APTT",min_value=0,max_value=180,value=80)
    FDP  =st.slider("FDP",min_value=0,max_value=80,value=30)
    D_Dimmer = st.slider("D-Dimmer",min_value=0,max_value=40,value=4)
    data = {
        "Platelet":Platelet,
        "Albumin":Albumin,
        "Cholesterol":Cholesterol,
        "Sodium":Sodium,
        "APTT":APTT,
        "FDP":FDP,
        "D-Dimmer":D_Dimmer,
    }

    features = pd.DataFrame(data,index=[0])

    st.form_submit_button("Submit for prediction")


col1, col2, col3,col4,col5,col6,col7 = st.columns(7)



with col1:
    st.metric("Platelet",str(features.iloc[:,0][0]))
with col2:
    st.metric("Albumin",str(features.iloc[:,1][0]))
with col3:
    st.metric("Cholesterol",str(features.iloc[:,2][0]))
with col4:
    st.metric("Sodium",str(features.iloc[:,3][0]))
with col5:
    st.metric("APTT",str(features.iloc[:,4][0]))
with col6:
    st.metric("FDP",str(features.iloc[:,5][0]))
with col7:
    st.metric("D-Dimmer",str(features.iloc[:,6][0]))


model_path = os.getcwd()+"/"+"RF.joblib"

print(model_path)


clf = load(model_path)
prediciotn = clf.predict(features)
prediciotn_prob = clf.predict_proba(features)

prob = float(prediciotn_prob[0][1])
prob = round(prob,3)

if  prob <= 0.35:
    st.success("The probability of Pulmonary embolism probability is about {} %".format(prob*100))
elif prob>0.35 and prob <=0.65:
    st.warning("The probability of Pulmonary embolism probability is about {} %".format(prob*100))
else:
    st.error("The probability of Pulmonary embolism probability is about {} %".format(prob*100))

st.subheader(r"The Decision Plot of Pulmonary Embolism")

explainer = shap.TreeExplainer(clf)
features_columns = features.columns.to_list()
shap_values_feature = explainer.shap_values(features)
expected_value0 = explainer.expected_value

fig1, ax1 = plt.subplots()
shap.decision_plot(expected_value0[1], shap_values_feature[1], features)
st.pyplot(fig1)