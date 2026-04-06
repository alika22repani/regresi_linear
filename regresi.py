import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('data_jk.csv')

print("ISI UMUR:", df['umur'].unique())

plt.style.use('default')

daftar_umur = df['umur'].unique()

for umur in daftar_umur:
    print("Diproses umur:", umur)

    df_filter = df[df['umur'] == umur]

    df_group = df_filter.groupby(['tahun','jenis_kelamin'])['jumlah_penduduk'].sum().reset_index()

    for jk in ['LAKI-LAKI', 'PEREMPUAN']:
        data = df_group[df_group['jenis_kelamin'] == jk]

        if data.empty:
            continue

        # fitur
        data['tahun_ke'] = data['tahun'] - data['tahun'].min()

        X = data[['tahun_ke']]
        Y = data['jumlah_penduduk']

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, Y)

        # range
        tahun_range = pd.DataFrame({
            'tahun_ke': range(int(X['tahun_ke'].min()), int(X['tahun_ke'].max()) + 5)
        })

        X_range_poly = poly.transform(tahun_range)
        Y_line = model.predict(X_range_poly)

        tahun_asli = tahun_range['tahun_ke'] + data['tahun'].min()

        # 🔥 scatter
        plt.scatter(data['tahun'], Y, s=20)

        # 🔥 PLOT (INI YANG BENAR)
        if jk == 'LAKI-LAKI':
            plt.plot(tahun_asli, Y_line, color='#4A90E2', label=f'L {umur}')
        else:
            plt.plot(tahun_asli, Y_line, linestyle='dashed', color='#FF6F91', label=f'P {umur}')

# =========================
# FINAL
# =========================
plt.title('Analisis Regresi Jumlah Penduduk Berdasarkan Umur dan Jenis Kelamin')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Penduduk')
plt.legend(fontsize=6, ncol=2, loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True)
plt.show()