# 📈 Kapsamlı Regresyon Analizi: Sigorta Maliyeti Tahmini

## 🎯 Projenin Amacı
Bu proje, bireylerin demografik ve sağlık özelliklerine (yaş, vücut kitle indeksi, sigara kullanımı vb.) bakarak sağlık sigortası maliyetlerini tahmin etmeyi amaçlamaktadır. Projenin ana odak noktası, sadece temel bir regresyon modeli kurmak değil; **Polinom Regresyon** ile eğrisel ilişkileri yakalamak ve **Düzenlileştirme (Regularization)** teknikleri ile modelin aşırı öğrenmesini (Overfitting) engellemektir.

## 🛠️ Kullanılan Teknolojiler ve Modeller
* **Veri İşleme & Görselleştirme:** Pandas, NumPy, Matplotlib, Seaborn
* **Makine Öğrenmesi (Scikit-Learn):**
  * `LinearRegression` (Çoklu Doğrusal Regresyon)
  * `PolynomialFeatures` (Polinom Regresyon Dönüşümü)
  * `Ridge` (L2 Regularization)
  * `Lasso` (L1 Regularization - Özellik Seçimi)
  * `ElasticNet` (L1 + L2 Regularization Kombinasyonu)

## 🚀 Proje Adımları
1. **Keşifçi Veri Analizi (EDA):** Veri setindeki dağılımlar incelenmiş ve özellikler (features) arasındaki korelasyonlar ısı haritaları (heatmap) ile analiz edilmiştir.
2. **Kategorik Veri Dönüşümü:** Makine öğrenmesi algoritmalarının çalışabilmesi için metinsel veriler (Örn: Sigara içme durumu) sayısal değerlere dönüştürülmüştür (Encoding).
3. **Model Karşılaştırmaları:** Veri seti üzerinde sırasıyla Linear, Polynomial, Ridge, Lasso ve ElasticNet algoritmaları eğitilmiştir.
4. **Regularization (Düzenlileştirme) Analizi:** Karmaşık modellerde ortaya çıkabilen ezberleme (Overfitting) sorunu Ridge ve Lasso'nun ceza parametreleri (Alpha/Penalty) kullanılarak dengelenmiştir.

## 📊 Öne Çıkan Öğrenimler
Bu proje, farklı regresyon algoritmalarının aynı veri seti üzerindeki performanslarını (R², MSE, MAE metrikleriyle) karşılaştırma ve hangi durumlarda L1 (Lasso) veya L2 (Ridge) cezalandırmalarının kullanılması gerektiğini anlamak açısından harika bir "Model Seçimi" (Model Selection) pratiği olmuştur.

[EN]

# 📈 Comprehensive Regression Analysis: Medical Cost Prediction

## 🎯 Objective
This project aims to predict individual medical insurance costs based on demographic and health-related features. The primary focus is not only to build a baseline predictive model but to explore non-linear relationships using **Polynomial Regression** and prevent model overfitting by applying advanced **Regularization** techniques.

## 🛠️ Technologies & Algorithms Used
* **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning Models (Scikit-Learn):**
  * `LinearRegression` (Multiple Linear Regression)
  * `PolynomialFeatures` (Polynomial Transformations)
  * `Ridge Regression` (L2 Penalty)
  * `Lasso Regression` (L1 Penalty & Feature Selection)
  * `ElasticNet` (Combination of L1 and L2 Penalties)

## 🚀 Workflow & Implementation
1. **Exploratory Data Analysis (EDA):** Conducted correlation analysis and visualized data distributions to identify key cost drivers (e.g., smoking status, BMI).
2. **Data Preprocessing:** Handled categorical encoding and applied Feature Scaling to ensure fair weight distribution among variables.
3. **Algorithm Comparison:** Trained and tested multiple regression algorithms side-by-side to observe how they handle the dataset's variance.
4. **Applying Regularization:** Utilized Ridge, Lasso, and ElasticNet to penalize large coefficients, effectively reducing the risk of overfitting in higher-degree polynomial transformations.

## 📊 Key Takeaways
This project serves as a practical demonstration of **Model Selection and Tuning**. By comparing standard linear models with regularized ones (Ridge/Lasso), it highlights the importance of managing bias-variance tradeoffs using R-Squared (R²), Mean Absolute Error (MAE), and Mean Squared Error (MSE) metrics.gi