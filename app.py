import joblib
import pickle

from flask import Flask, render_template, request

# # Opsi 1 : Opsi Klasik - Opsi Statis
# - Dataset
# df.to_html("dataset.html")

# - Visualisasi
# plt.savefig

# - Prediction
# Model Import 

app = Flask(__name__)

@app.route('/')
def home():
    #return 'Selamat Datang'
    return render_template('home.html')

@app.route('/visualize')
def vis():
    return render_template('viz.html')

@app.route('/predict')
def pred():
    return render_template('pred.html')

@app.route('/hasil', methods=['POST'])
def result():
    if request.method == 'POST':
        input = request.form
        sl = float(input["SL"])
        sw = float(input["SW"])
        pl = float(input["PL"])
        pw = float(input["PW"])
        prediksi = Model.predict([[sl, sw, pl, pw]])[0]
    return render_template('result.html', data = input, pred = prediksi)

if __name__ == "__main__":
    Model = joblib.load("Model_RF_Tuned")
    #Model = pickle.load(open("Model_Iris_DT_2.pkl", "rb"))
    app.run(debug=True)