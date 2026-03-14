# 🚀 Polinom Regresyon ve Scikit-Learn Pipeline Mimarisi

## 🎯 Projenin Amacı
Bu mini proje, doğrusal (linear) olmayan veri setlerini daha yüksek dereceli denklemlerle modelleyebilmek için **Polinom Regresyon (Polynomial Regression)** kullanımını uygulamalı olarak göstermektedir. Ayrıca, makine öğrenmesi süreçlerini otomatize etmek, kod karmaşasını önlemek ve veri sızıntısının (data leakage) önüne geçmek amacıyla Scikit-Learn **Pipeline** mimarisi kullanılmıştır.

*(Senaryo: Bir sporcunun antrenman saati ile gelişim skoru arasındaki eğrisel (non-linear) ilişkinin modellenmesi.)*

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python
* **Veri İşleme:** Pandas, NumPy
* **Makine Öğrenmesi (Scikit-Learn):** * `Pipeline` (Veri akış yönetimi)
  * `PolynomialFeatures` (Özellik mühendisliği / Dönüşüm)
  * `StandardScaler` (Veri ölçeklendirme)
  * `LinearRegression` (Temel model)
* **Görselleştirme:** Matplotlib (Modelin eğrisel uyumunu görselleştirmek için)

## 🚀 Proje Adımları ve Pipeline İşleyişi
1. **Veri Hazırlığı:** Bağımsız ve bağımlı değişkenler arasındaki doğrusal olmayan ilişki tanımlanmış ve veri seti `train_test_split` ile ayrılmıştır.
2. **Pipeline Kurulumu:** Modelin eğitim ve test aşamalarındaki adımları tek bir çatı altında toplanmıştır:
   * **Adım 1:** Veriye `PolynomialFeatures(degree=3)` uygulanarak özelliklerin derecesi artırılmıştır.
   * **Adım 2:** Veriler `StandardScaler` ile ölçeklendirilmiştir.
   * **Adım 3:** Dönüştürülmüş ve ölçeklendirilmiş veri `LinearRegression` modeline beslenmiştir.
3. **Model Eğitimi:** Tek bir `.fit()` metodu ile tüm ardışık işlemler eğitim setine uygulanmıştır. Test setine ise sadece `.predict()` uygulanarak dönüşümlerin otomatik yapılması sağlanmış, manuel hataların önüne geçilmiştir.
4. **Değerlendirme:** Modelin eğitim ve test setlerindeki R-Squared (R²) skorları karşılaştırılarak, `degree` (derece) parametresinin Overfitting (Aşırı Öğrenme) ve Underfitting (Eksik Öğrenme) üzerindeki etkisi analiz edilmiştir.

## 📊 Öne Çıkan Öğrenimler
Bu proje, karmaşık makine öğrenmesi süreçlerinin Pipeline nesneleri ile nasıl temiz ve sürdürülebilir bir koda (clean code) dönüştürüleceğini kanıtlamaktadır.

[EN]

# 🚀 Polynomial Regression & Scikit-Learn Pipeline Architecture

## 🎯 Objective
This mini-project practically demonstrates the use of **Polynomial Regression** to model datasets with non-linear relationships. Furthermore, it showcases the implementation of the Scikit-Learn **Pipeline** architecture to automate the machine learning workflow, write clean code, and rigorously prevent data leakage during preprocessing.

*(Scenario: Modeling the non-linear/curved relationship between training hours and a performance development score.)*

## 🛠️ Technologies & Modules
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning (Scikit-Learn):**
  * `Pipeline` (Workflow automation)
  * `PolynomialFeatures` (Feature transformation)
  * `StandardScaler` (Feature scaling)
  * `LinearRegression` (Estimator)
* **Visualization:** Matplotlib (To visualize the polynomial curve fitting)

## 🚀 Workflow & Pipeline Operations
1. **Data Preparation:** The dataset, exhibiting a non-linear trend, was generated and split using `train_test_split`.
2. **Pipeline Construction:** Preprocessing and modeling steps were chained into a single object:
   * **Step 1:** `PolynomialFeatures(degree=3)` was applied to generate higher-order terms.
   * **Step 2:** `StandardScaler` was used to normalize the transformed features.
   * **Step 3:** The processed data was fed into a `LinearRegression` model.
3. **Execution:** By calling the `.fit()` method on the Pipeline, all sequential transformations and model training were executed simultaneously on the training set. The test set was seamlessly transformed during the `.predict()` phase, eliminating manual transformation errors.
4. **Evaluation:** R-Squared (R²) scores on both training and test sets were compared to analyze the impact of the polynomial `degree` hyperparameter on Overfitting and Underfitting.

## 📊 Key Takeaway
This repository highlights the best practice of using Pipelines in machine learning to ensure code readability, maintainability, and robust model evaluation.