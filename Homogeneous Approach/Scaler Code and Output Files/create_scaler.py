import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob
import os

print("Global scaler oluşturma işlemi başlatıldı...")

# G2: Tüm client'lar tarafından kullanılan özelliklerin birleşim kümesi (Union of features)
# Client 5'in listesi tüm özellikleri içerdiği için onu temel alıyoruz.
ALL_POSSIBLE_FEATURES = [
    "PACK_V_SUM_OF_CELLS",
    "PACK_I_HALL",
    "CELL_T_MAX_VAL",
    "vehicle_speed"
]

# Hedef değişkenimiz
TARGET_FEATURE = "PACK_Q_SOC_INTERNAL"

# Tüm client veri dosyalarının yollarını bul
data_path = r"C:\Users\Erncl\Desktop\ercodex\Projects_Eren_Coding\ICT Summer School\Federated Learning Mini Project"
all_files = glob.glob(os.path.join(data_path, "Client*.csv"))

if not all_files:
    print(f"Hata: '{data_path}' klasöründe 'Client*.csv' formatında dosya bulunamadı.")
    exit()

print(f"Bulunan veri dosyaları: {[os.path.basename(f) for f in all_files]}")

# Standartlaştırılmış DataFrame'leri tutacak liste
standardized_dfs = []

# Her bir dosyayı oku ve işle
for filename in all_files:
    try:
        df = pd.read_csv(filename)
        
        # O anki client verisinde eksik olan sütunları bul ve 0 ile doldur
        for col in ALL_POSSIBLE_FEATURES:
            if col not in df.columns:
                # print(f"'{os.path.basename(filename)}' dosyasında '{col}' sütunu eksik. Sıfır ile dolduruluyor.")
                df[col] = 0
        
        standardized_dfs.append(df)
    except Exception as e:
        print(f"Hata: '{filename}' dosyası okunurken bir sorun oluştu: {e}")
        exit()

# Tüm client verilerini tek bir DataFrame'de birleştir
master_df = pd.concat(standardized_dfs, axis=0, ignore_index=True)

print(f"\nTüm veriler birleştirildi. Toplam satır sayısı: {len(master_df)}")

# G10: SoC'nin bir önceki adimini (t-1) özellik olarak kullan
master_df['PACK_Q_SOC_INTERNAL_t-1'] = master_df[TARGET_FEATURE].shift(1)
master_df = master_df.dropna().reset_index(drop=True) # Shift'ten kaynaklanan NaN satırını kaldır

# Özellik listesine t-1 sütununu da ekle
FEATURES_FOR_SCALING = ALL_POSSIBLE_FEATURES + ["PACK_Q_SOC_INTERNAL_t-1"]

# Özellikler (X) ve hedef (y) olarak ayır
X = master_df[FEATURES_FOR_SCALING]
y = master_df[[TARGET_FEATURE]]

# X (özellikler) için scaler oluştur ve kaydet
scaler_x = MinMaxScaler()
scaler_x.fit(X)
joblib.dump(scaler_x, 'scaler_x.pkl')
print("'scaler_x.pkl' başarıyla oluşturuldu ve kaydedildi.")

# y (hedef) için scaler oluştur ve kaydet
scaler_y = MinMaxScaler()
scaler_y.fit(y)
joblib.dump(scaler_y, 'scaler_y.pkl')
print("'scaler_y.pkl' başarıyla oluşturuldu ve kaydedildi.")

print("\nİşlem başarıyla tamamlandı.")