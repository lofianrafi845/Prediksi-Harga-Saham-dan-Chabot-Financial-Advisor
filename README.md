# 📈 Prediksi Harga Saham & Chatbot Financial Advisor

Selamat datang di aplikasi **Prediksi Harga Saham IDX** berbasis AI! Aplikasi ini hadir sebagai co-pilot Anda dalam menavigasi pasar saham Indonesia — mengubah data kompleks menjadi prediksi yang mudah dipahami langsung di browser.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediksi-harga-saham-dan-chabot-financial-advis.streamlit.app)

---

## 🚀 Tentang Proyek

**StockMaster AI** adalah aplikasi web berbasis Streamlit untuk memprediksi harga saham IDX (Bursa Efek Indonesia). Tujuannya sederhana: memberikan perspektif berbasis data untuk membantu keputusan investasi Anda.

Fitur utama:
- 📊 **Prediksi Harga** menggunakan model XGBoost & LSTM terlatih
- 💬 **Financial Advisor Chatbot** berbasis RAG + Groq LLM (LLaMA 3.3)
- 📉 **Visualisasi Interaktif** dengan chart dark-mode bergaya modern
- 📋 **Analisis Fundamental** (ROE, ROA, DER, EPS) per saham IDX

---

## 🛠️ Cara Memulai

### Opsi 1: Akses Aplikasi Langsung (Direkomendasikan)
Tidak perlu instalasi! Klik link berikut untuk langsung menggunakan aplikasi:

👉 **[Buka Aplikasi di sini](https://prediksi-harga-saham-dan-chabot-financial-advis.streamlit.app)**

---

### Opsi 2: Jalankan Secara Lokal

**1. Clone Repository**
```bash
git clone https://github.com/lofianrafi845/Prediksi-Harga-Saham-dan-Chabot-Financial-Advisor.git
cd Prediksi-Harga-Saham-dan-Chabot-Financial-Advisor
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Training Model** *(hanya perlu dilakukan sekali)*
```bash
python train_models.py
```
Perintah ini akan melatih model XGBoost dari data CSV dan menyimpannya ke folder `model/`.

**4. Jalankan Aplikasi**
```bash
streamlit run app.py
```

---

## 📁 Struktur Proyek

```
├── app.py                          # Aplikasi Streamlit utama
├── train_models.py                 # Script training model
├── requirements.txt                # Dependensi Python
├── stockmaster_preprocessed.csv    # Dataset saham IDX
├── model/
│   ├── xgb_model.pkl              # Model XGBoost terlatih
│   ├── scaler.pkl                 # MinMaxScaler
│   └── model_metrics.pkl          # Metrik evaluasi model
└── Modelling.ipynb                 # Notebook eksplorasi & training
```

---

## ⚙️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Frontend | Streamlit |
| Prediksi | XGBoost, LSTM (scikit-learn, TensorFlow) |
| Chatbot | Groq API (LLaMA 3.3 70B) |
| Visualisasi | Plotly |
| Data | yfinance, pandas |

---

## 📊 Dataset

Dataset berisi harga saham harian dari **50 saham IDX** mulai Oktober 2024 – Juni 2026, dengan fitur:
- `Close_Price` — Harga penutupan
- `Basic EPS`, `DER`, `ROA`, `ROE` — Indikator fundamental
- `MA5`, `MA20` — Moving Average
- `Volatility20` — Volatilitas 20-hari
- `Target` — Harga penutupan hari berikutnya (label)

---

## ⚠️ Disclaimer

> Analisis yang disediakan aplikasi ini bersifat **informatif** dan **bukan merupakan saran investasi resmi**. Selalu lakukan riset mendalam dan konsultasikan dengan penasihat keuangan sebelum membuat keputusan investasi.
