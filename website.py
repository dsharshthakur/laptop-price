import streamlit as slt
import pickle
import pandas as pd
import numpy as np

file=open("dataframe (1).pkl","rb")
df=pickle.load(file)
df.drop("Price",axis=1,inplace=True)
print(df.columns)

#model
file=open("model (1).pkl", "rb")
model=pickle.load(file)
slt.header("Search for the best Laptop!!")


#brand name
laptop_brand=slt.selectbox(label="select company",options=df["Company"].unique())

#type
laptop_type=slt.selectbox(label="laptop type",options=df["TypeName"].unique())

#Ram size
ram=slt.selectbox("Ram(in GB)",options=[2,4,6,8,12,16,24,32,64])

#screen size
screen_size=slt.number_input("Screen Size (in inches.)",min_value=5)

#screen resolution
screen=slt.selectbox(label="Screem Resolution",options=['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
x_resolution=int(screen.split("x")[0])
y_resolution=int(screen.split("x")[1])

#ppi
ppi=np.sqrt(x_resolution**2+y_resolution**2)/screen_size

#weight
weight=slt.number_input(label="laptop weight(in kg)")

#touch screen
touch=slt.selectbox(label="Touch Screen",options=["Yes","No"])
if touch=="Yes":
    touch=1
else:
    touch=0
#ips
ips=slt.selectbox(label="ips",options=["Yes","No"])
if ips=="Yes":
    ips=1
else:
    ips=0

#processor brand
processor_brand=slt.selectbox(label="Processor",options=df["processor brand"].unique())

#HDD
hdd=slt.selectbox(label="HDD",options=[0,8,32,64,128,256,512,1024])

#SSD
ssd=slt.selectbox(label="SSD",options=[0,8,32,64,128,256,512,1024])

#gpu brand
gpu_brand=slt.selectbox(label="Gpu",options=df["gpu brand"].unique())

#operating system
os=slt.selectbox(label="Operating system",options=df["operating system"].unique())

predict_btn=slt.button(label="Predict Price")

if predict_btn==True:
    row=np.array([laptop_brand,laptop_type,ram,weight,touch,ips,ppi,processor_brand,ssd,hdd,gpu_brand,os])
    data_dict={
        "Company":[laptop_brand],"TypeName":[laptop_type],"Ram":[ram],"Weight":[weight],"touch screen":[touch],"ips or not":[ips],
        "pixel per inch":[ppi],"processor brand":[processor_brand],"SSD":[ssd],"HDD":[hdd],"gpu brand":[gpu_brand],"operating system":[os]
    }
    new_df=pd.DataFrame(data_dict)
    prediction=model.predict(new_df)
    prediction_without_log=np.exp(prediction)[0]
    slt.text("Price:")
    slt.subheader(round(prediction_without_log))
print(df)
