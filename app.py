import joblib
import pickle
import pandas as pd

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

@app.route('/predict')
def pred():
    return render_template('pred.html')

@app.route('/hasil', methods=['POST'])
def result():
    if request.method == 'POST':
        input = request.form
        id1 = input["ID"]
        # print(id1)
        moa = input["MOA"]
        # print(moa)
        dotw = input["DOTW"]
        # print(dotw)
        te = input["TE"]
        # print(te)
        dfrtw = input["DFRTW"]
        # print(dfrtw)
        st = input["ST"]
        # print(st)
        age = input["AGE"]
        # print(age)
        wlad = input["WLAD"]
        # print(wlad)
        edu = input["EDU"]
        # print(edu)
        son = input["SON"]
        # print(son)
        sd = input["SD"]
        # print(sd)
        atin = input["ATIN"]
        # print(atin)
        # print(input)
        hasil = [[id1, moa, dotw, te, dfrtw, st, age, wlad, edu, son, sd, atin]]
        print(hasil)
        df = pd.DataFrame(data=hasil, columns=['ID','Month of absence','Day of the week','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Education','Son','Social drinker','Absenteeism time in hours'])
        print(df)
        prediksi = Model.predict(df)
    return render_template('result.html', data = input, pred = prediksi)

if __name__ == "__main__":
    Model = joblib.load("Model_RF_Tuned")
    #Model = pickle.load(open("Model_Iris_DT_2.pkl", "rb"))
    app.run(debug=True)