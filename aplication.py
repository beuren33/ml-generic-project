from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from src.Pipeline.predict_pipeline import CustomData,PredictionPipeline

aplication = Flask(__name__)

@aplication.route('/')
def index():
    return render_template('index.html')

@aplication.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            white_rating=int(request.form.get('white_rating')),
            black_rating=int(request.form.get('black_rating')),
            opening_eco=request.form.get('opening_eco'),
        )

    pred_df=data.get_data_frame()
    
    predict_pipeline=PredictionPipeline()
    results=predict_pipeline.predict(pred_df)

    prob_black= round(results[0][0] * 100,2)
    prob_white= round(results[0][1] * 100,2)
    prob_draw= round(results[0][2] * 100,2)

    return render_template('home.html',prob_black=prob_black,prob_white=prob_white,prob_draw=prob_draw)

if __name__=="__main__":
    aplication.run(host="0.0.0.0",port=8080)
    
