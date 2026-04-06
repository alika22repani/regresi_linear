from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

df = pd.read_csv('data_jk.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    daftar_umur = df['umur'].unique()
    hasil_l = None
    hasil_p = None
    grafik = None

    if request.method == 'POST':
        umur_pilih = request.form['umur']
        tahun_input = int(request.form['tahun'])

        df_filter = df[df['umur'] == umur_pilih]
        df_group = df_filter.groupby(['tahun','jenis_kelamin'])['jumlah_penduduk'].sum().reset_index()

        plt.figure(figsize=(8,5))

        for jk in ['LAKI-LAKI', 'PEREMPUAN']:
            data = df_group[df_group['jenis_kelamin'] == jk]

            if data.empty:
                continue

            data['tahun_ke'] = data['tahun'] - data['tahun'].min()

            X = data[['tahun_ke']]
            Y = data['jumlah_penduduk']

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, Y)

            # ======================
            # PREDIKSI TAHUN INPUT
            # ======================
            tahun_ke_input = tahun_input - data['tahun'].min()
            X_pred = poly.transform([[tahun_ke_input]])
            hasil = model.predict(X_pred)[0]

            if jk == 'LAKI-LAKI':
                hasil_l = hasil
            else:
                hasil_p = hasil

            # ======================
            # GRAFIK
            # ======================
            tahun_range = pd.DataFrame({
                'tahun_ke': range(int(X['tahun_ke'].min()), int(X['tahun_ke'].max()) + 5)
            })

            X_range_poly = poly.transform(tahun_range)
            Y_line = model.predict(X_range_poly)

            tahun_asli = tahun_range['tahun_ke'] + data['tahun'].min()

            plt.scatter(data['tahun'], Y, s=20)

            if jk == 'LAKI-LAKI':
                plt.plot(tahun_asli, Y_line, color='#4A90E2', label='Laki-laki')
            else:
                plt.plot(tahun_asli, Y_line, linestyle='dashed', color='#FF6F91', label='Perempuan')

        plt.title(f'Prediksi Umur {umur_pilih}')
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Penduduk')
        plt.legend()
        plt.grid(True)

        # simpan grafik
        if not os.path.exists('static'):
            os.makedirs('static')

        path = 'static/grafik.png'
        plt.savefig(path)
        plt.close()

        grafik = path

    return render_template(
        'index.html',
        umur=daftar_umur,
        hasil_l=hasil_l,
        hasil_p=hasil_p,
        grafik=grafik
    )

if __name__ == '__main__':
    app.run(debug=True)