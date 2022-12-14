from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle 
import joblib


model = pickle.load(open('DT.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
encoding = joblib.load('encoding')
scale = joblib.load('minmax')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/success',methods = ['POST'])
def success():
    if request.method == "POST":
        f = request.files['file']
        data_new = pd.read_excel(f)
        #data2 = copy.deepcopy(data)
        #imputation
        labour_new1 = data_new.drop(['Emp ID','Name','Nationality','Martial Status','Designation','Experience','Work Date','Site','Rfid','Latitude','Longitude','Work Started Time','Noise Detection','Gas Sensor'], axis=1)
        numeric_features = labour_new1.select_dtypes(exclude=['object']).columns
        
        #numeric_features
        
        #categorical_features = labour_new1.select_dtypes(include=['object']).columns
        
        #categorical_features
        
        impute1 = pd.DataFrame(impute.transform(labour_new1),columns=numeric_features)
        
        impute1[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']] = winsor.transform(impute1[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']])
        
        clean2=pd.DataFrame(scale.transform(impute1))
        clean3=pd.DataFrame(encoding.transform(labour_new1).todense())
        clean_data=pd.concat([clean2,clean3],axis=1,ignore_index=True)
        prediction=pd.DataFrame(model.predict(clean_data),columns=['Performance'])
        final_data=pd.concat([prediction, labour_new1],axis=1)
        
        return render_template('data.html', Y = final_data.to_html(justify='center'))
    
    
if __name__=='__main__':
    app.run(debug = True)
