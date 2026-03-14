# 📈 Lineer Regresyon ile Tahmin Modeli (Linear Regression)

## 🎯 Projenin Amacı
Bu mini proje, bağımsız değişkenler (özellikler) ile sürekli bir bağımlı değişken (hedef) arasındaki doğrusal ilişkiyi modellemek amacıyla geliştirilmiştir. Amaç, veri setindeki örüntüleri öğrenerek gelecekteki veriler için sayısal tahminler (örneğin: fiyat, maliyet, maaş) üretebilen bir makine öğrenmesi modeli kurmaktır.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Dil:** Python
* **Veri İşleme:** Pandas, NumPy
* **Görselleştirme:** Matplotlib, Seaborn
* **Makine Öğrenmesi:** Scikit-Learn (LinearRegression, train_test_split, StandardScaler)

## 🚀 Proje Adımları
1. **Keşifçi Veri Analizi (EDA):** Veri seti incelenmiş, eksik değerler kontrol edilmiş ve değişkenler arasındaki korelasyonlar görselleştirilmiştir.
2. **Veri Ön İşleme:** Modelin algoritmik başarısını artırmak için veriler eğitim (train) ve test (test) setlerine ayrılmış, gerekli ölçeklendirme (Scaling) işlemleri yapılmıştır.
3. **Model Kurulumu:** Scikit-Learn kütüphanesi kullanılarak Çoklu Doğrusal Regresyon (Multiple Linear Regression) modeli eğitilmiştir.
4. **Değerlendirme:** Modelin tahmin başarısı **R-Squared (R²)**, **Mean Absolute Error (MAE)** ve **Mean Squared Error (MSE)** metrikleri kullanılarak ölçülmüştür.

## 📊 Sonuçlar
Kurulan Lineer Regresyon modeli, test verileri üzerinde çalıştırılmış ve elde edilen hata metrikleri, değişkenler arasındaki doğrusal ilişkinin anlamlı bir seviyede modellendiğini göstermiştir.


[EN]


# 📈 Predictive Modeling with Linear Regression

## 🎯 Objective
This mini-project was developed to model the linear relationship between independent variables (features) and a continuous dependent variable (target). The primary goal is to build a machine learning model capable of learning patterns from the dataset to generate numerical predictions (e.g., cost, price, salary) for unseen data.

## 🛠️ Technologies & Libraries Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (LinearRegression, train_test_split, StandardScaler)

## 🚀 Workflow & Steps
1. **Exploratory Data Analysis (EDA):** The dataset was analyzed to understand data distributions, handle missing values, and visualize correlations between variables.
2. **Data Preprocessing:** The dataset was split into training and testing sets. Feature scaling was applied where necessary to improve algorithm efficiency.
3. **Model Training:** A Multiple Linear Regression model was built and trained using the Scikit-Learn library.
4. **Model Evaluation:** The performance of the predictive model was evaluated using industry-standard metrics such as **R-Squared (R²)**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**.

## 📊 Results & Conclusion
The trained Linear Regression model was tested on unseen data. The evaluation metrics demonstrate a statistically significant linear relationship between the features and the target variable, making the model reliable for basic predictive tasks.