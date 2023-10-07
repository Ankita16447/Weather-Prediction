from flask import Flask , render_template , request
import pickle
import numpy as np

model = pickle.load(open('weather.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/submit' , methods = ['POST'])
def home():
    # if request.method == "POST":
    minTemp = request.form['min_temp']
    maxTemp = request.form['max_temp']
    rainFall = request.form['rainfall']
    Evaporation = request.form['evaporation']
    sunShine = request.form['sunshine']
    windSpeed = request.form['wind_speed']
    humidity = request.form['humidity9']
    Humidity = request.form['humidity3']
    pressure = request.form['pressure9']
    Pressure = request.form['pressure3']
    cloud = request.form['cloud9']
    Cloud = request.form['cloud3']
    temp = request.form['temp9']
    Temp = request.form['temp3']
    RainToday = request.form['rainT']
    rainMM = request.form['rain_mm']

    arr = np.array([[1, minTemp , maxTemp , rainFall,Evaporation,windSpeed,sunShine, humidity,Humidity,pressure,Pressure,cloud , Cloud , temp , Temp , RainToday , rainMM]])
    arr = np.array(arr, dtype = float)

    pred = model.predict(arr)


    return render_template('submit.html' , data = pred)


if __name__ =="__main__":
    app.run(debug=True)
