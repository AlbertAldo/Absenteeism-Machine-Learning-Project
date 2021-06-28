# Absenteeism-Machine-Learning-Project

# Define Problems :
- Sering kali produktivitas dari perusahaan terganggu dikarenakan banyaknya Karyawan yang Absen/Tidak masuk kerja dan cenderung membuat performa perusahaan menurun. Maka perusahaan akan mengalami kerugian dikarenakan hal tersebut, sehingga perlu diantisipasi agar mengurangi kecenderungan Karyawan melakukan Absen/Tidak masuk kerja untuk Alasan yang sebenarnya masih bisa ditoleransi.

# Goals :
- Membuat model untuk memprediksi Alasan Karyawan yang ingin Absen/Tidak masuk kerja.
- Dengan mengetahui hal tersebut, sehingga dikemudian hari jika ada karyawan yang melakukan Absen/Tidak masuk kerja untuk Alasan yang sebenarnya masih bisa ditoleransi, perusahaan dapat melakukan penanganan terhadap karyawan tersebut.

# Handling Outliers :
#### Berdasarkan boxplot di atas dapat dilihat bahwa :
- Transportation expense memiliki 1 nilai outliers
- Service time memiliki 1 nilai outliers
- Age memiliki 1 nilai outliers
- Work load Average/day memiliki 1 nilai outliers
- Hit target memiliki 1 nilai outliers
- Pet memiliki 3 nilai outliers
- Height memiliki 5 nilai outliers
- Absenteeism time in hours memiliki 10 nilai outliers

#### Untuk nilai-nilai outliers diatas tidak dilakukan handling agar tidak mengubah data apapun pada kasus ini.

# Feature Selection untuk Machine Learning
```
dfbaru = df.drop(columns=['Reason for absence', 'Seasons', 'Hit target', 'Disciplinary failure', 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index', 'BMI', 'Distance', 'Ongkos', 'Stress'])
```

Drop Kolom yang tidak digunakan untuk Machine Learning :
- Reason for absence dikarenakan sudah diganti menjadi Without ICD dan ICD.
- Season dikarenakan sudah diwakilkan oleh Month of absence.
- Hit target dikarenakan tidak ada hubungan absen dengan Hit target berdasarkan EDA.
- Disciplinary failure dikarenakan semua karyawan yang memiliki Disciplinary yang baik.
- Social smoker dikarenakan tidak ada hubungan absen dengan Social smoker berdasarkan EDA.
- Pet dikarenakan tidak ada hubungan absen dengan Pet berdasarkan EDA.
- Weight, Height, Body mass index dikarenakan BMI tidak ada hubungan dengan absen berdasarkan EDA.
- Distance, Ongkos, Stress dikarenakan hanya untuk kolom bantu analisa EDA.

# Machine Learning

Menggunakan Model RandomForestClassifier didapatkan hasil sebagai berikut :

Data  | Accuracy | Recall | Precision | F1-Score
-----|------|------|------|-----|
Training Pipeline RandomForestClassifier Tuned | 0.800718   |0.876437	 | 0.817694 | 0.846047	
Test Pipeline RandomForestClassifier Tuned | 0.785714	   | 0.862069 | 0.806452 | 0.846047

Dengan hasil classification report sebagai berikut :

![ClassificationReport](https://i.imgur.com/0bzqVjf.png)

### Penjelasan Classification Report
- Precision 0 dan Recall 1 searah, jika nilai Precision 0 naik maka nilai Recall 1 juga akan naik, begitu juga sebaliknya.
- Precision 1 dan Recall 0 searah, jika nilai Precision 1 naik maka nilai Recall 0 juga akan naik, begitu juga sebaliknya.
- Precision 0 dan Precision 1 berbanding terbalik, jika nilai Precision 0 naik maka nilai Precision 1 akan turun, begitu juga sebaliknya.
- Recall 0 dan Recall 1 berbanding terbalik, jika nilai Recall 0 naik maka nilai Recall 1 akan turun, begitu juga sebaliknya.
- Accuracy dapat digunakan ketika dataset Balance.
- Accuracy merupakan perbandingan antara Jumlah Seluruh Tebakan / Prediksi yang Benar (True) dibandingkan dengan Jumlah Seluruh Data / Tebakan / Prediksi.
- Precision menargetkan False Positive sekecil mungkin. (Ini diabaikan)
- Recall menargetkan False Negative sekecil mungkin. (Ini diabaikan)

Fokus Model ini pada **Accuracy** dikarenakan pada Kolom Target **(Reason)** valuenya masing-masing **balance**. Model terbaik menghasilkan **Nilai Accuracy 0.79** pada perbandingan y_test dengan y_test_RF_Tuned dengan menggunakan **(Model Random Forest Classifier dengan Hyper Parameter Tuning) dan tidak terlihat Overfit dari segi Nilai data Train & Test**.

# Conclusion
- Untuk Alasan Karyawan Absen/Tidak Masuk Kerja yang termasuk golongan Without ICD seharusnya dapat ditoleransi oleh Karyawan tersebut, sehingga karyawan tidak perlu Absen/Tidak masuk Kerja. Absen/Tidak Masuk kerja yang wajar adalah untuk Karyawan yang sedang sakit dirawat atau terkena Disease, atau ada Sanak/Saudara yang meninggal. 
- Ada sebesar 62,4% Karyawan yang melakukan Absen/Tidak Masuk Kerja dengan Without ICD, dan 37,6% Karyawan yang melakukan Absen/Tidak Masuk Kerja dengan ICD.
- Paling banyak Karyawan Absen/tidak masuk kerja selama 8 jam.
- Karyawan yang rumah tinggalnya jaraknya dekat ke kantor paling jarang melakukan Absen/Tidak Masuk Kerja.
- Karyawan yang berlatar belakang pendidikan High School paling sering melakukan Absen/Tidak Masuk Kerja.
- Season, Hit target, Social smoker, Pet, Weight, Height, BMI dari data tidak mempengaruhi Karyawan untuk melakukan Absen/Tidak Masuk Kerja.
- Model terbaik yang digunakan untuk Machine Learning ini adalah Model Random Forest Classifier dengan Hyper Parameter Tuning yang menggunakan best_estimator_ dengan nilai Accuracy untuk data Train 0.800718 dan untuk data Test 0.785714.

# Recommendation
- Dengan adanya Machine Learning, Perusahaan dapat melakukan prediksi jika si Karyawan melakukan Absen/Tidak Masuk Kerja, apabila si Karyawan tetap melakukan Absen/Tidak Masuk Kerja dengan Reason **Without ICD** maka perusahaan dapat melakukan tindakan kepada si Karyawan tersebut.
- Dikarenakan banyaknya Karyawan yang sering Absen/Tidak Masuk Kerja dengan Reason **Without ICD** maka Perusahaan perlu mempertimbangkan untuk menyediakan Dokter Umum di Kantor, agar bagi para pekerja yang merasa untuk butuh Medical Consultation (Alasan Without ICD paling tinggi dikarenakan mereka Medical Consultation) dapat berkonsultasi dengan Dokter Umum di Kantor sehingga tidak perlu Absen/Tidak Masuk Kerja.
- Untuk Kategori yang termasuk **Without ICD** seperti Patient Follow Up, Medical Consultation, Blood Donation, Laboratory Examination, Physiotherapy, Dental Consultation bagi Perusahaan perlu disediakan Dokter/Tenaga Medis yang dapat menangani hal tersebut.
- Perusahaan harus memberikan penghargaan bagi Karyawan yang paling sedikit untuk Absen/Tidak Masuk Kerja, sehingga para Karyawan akan semakin semangat bekerja dan berlomba-lomba untuk menjadi yang paling sedikit Absen/Tidak Masuk Kerja.
