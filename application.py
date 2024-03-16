from flask import Flask,request,render_template
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application

# Map numerical prediction result to class label
def map_result_to_class(result):
    class_labels = {
        0: "Insufficient Weight",
        1: "Normal Weight",
        2: "Obesity Type-I",
        3: "Obesity Type-II",
        4: "Obesity Type-III",
        5: "Overweight Level-I",
        6: "Overweight Level-II"
        # Add mappings for other numerical results if needed
    }
    return class_labels.get(result, "Unknown")

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def classify_datapoint():
    if request.method == 'GET': 
        return render_template('form.html')
    else:
        data = CustomData(
            Gender=request.form.get('gender'),
            Age=float(request.form.get('age')),
            Height=float(request.form.get('height')),
            Weight=float(request.form.get('weight')),
            family_history_with_overweight=request.form.get('family_history'),
            FAVC=request.form.get('favc'),
            FCVC=float(request.form.get('fcvc')),
            NCP=float(request.form.get('ncp')),
            CAEC=request.form.get('caec'),
            SMOKE=request.form.get('smoke'),
            CH2O=float(request.form.get('ch2o')),
            SCC=request.form.get('scc'),
            FAF=float(request.form.get('faf')),
            TUE=float(request.form.get('tue')),
            CALC=request.form.get('calc'),
            MTRANS=request.form.get('mtrans')
        )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        class_label = map_result_to_class(results[0])  # Map numerical result to class label
        return render_template('form.html', results=class_label)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

