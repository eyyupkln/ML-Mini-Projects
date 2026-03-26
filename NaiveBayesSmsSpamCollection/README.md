# 📩 SMS Spam Sınıflandırma Modeli (Naive Bayes & NLP)

## 🎯 Projenin Amacı
Bu proje, makine öğrenmesi ve Doğal Dil İşleme (NLP) temellerini kullanarak gerçek dünyadan alınmış 5.500'den fazla SMS mesajının **Spam (İstenmeyen)** veya **Ham (Normal)** olup olmadığını sınıflandırmayı amaçlamaktadır. Proje, metin verilerinin makine öğrenmesi algoritmaları için nasıl sayısal formatlara dönüştürüleceğini (Bag of Words yaklaşımı) göstermektedir.

## 🛠️ Kullanılan Teknolojiler ve Modeller
* **Veri İşleme:** Pandas, NumPy
* **Doğal Dil İşleme (NLP):** `CountVectorizer` (Metinleri kelime frekanslarına dönüştürme)
* **Makine Öğrenmesi Modeli:** `MultinomialNB` (Çok Terimli Naive Bayes)
* **İş Akışı (Workflow):** `Pipeline` (Veri sızıntısını önlemek ve kod düzenini sağlamak için)

## 🚀 Proje Adımları
1. **Veri Hazırlığı:** 'SMS Spam Collection' veri seti projeye dahil edilmiş ve metin etiketleri (ham/spam), modelin anlayabilmesi için sayısal (0/1) değerlere çevrilmiştir.
2. **Özellik Çıkarımı (Feature Extraction):** Scikit-Learn kütüphanesindeki `CountVectorizer` aracı kullanılarak, metin mesajları içindeki kelimeler sayılmış ve "Kelime Çantası (Bag of Words)" matrisleri oluşturulmuştur.
3. **Model Eğitimi:** Kelime sayımları ve frekansları üzerinde en iyi çalışan algoritmalardan biri olan Naive Bayes (`MultinomialNB`) algoritması kullanılarak model eğitilmiştir.
4. **Değerlendirme:** Test seti üzerindeki başarı oranı `Accuracy Score` ile ölçülmüş; modelin doğru bildikleri ve yaptığı yanlış alarmlar (False Positives) detaylı bir `Confusion Matrix` (Hata Matrisi) ile analiz edilmiştir.

## 📊 Öne Çıkan Öğrenimler
Bu çalışma, yapılandırılmamış (unstructured) metin verilerinin temiz, okunabilir bir Pipeline mimarisiyle nasıl sınıflandırılabileceğini gösteren harika bir temel (baseline) NLP projesidir.

[EN]

# 📩 SMS Spam Classification (Naive Bayes & Basic NLP)

## 🎯 Objective
The goal of this project is to apply foundational Natural Language Processing (NLP) techniques and machine learning to classify over 5,500 real-world SMS messages as either **Spam** or **Ham (Normal)**. This project serves as a practical demonstration of converting raw text data into a machine-readable numerical format using the "Bag of Words" approach.

## 🛠️ Tech Stack & Libraries
* **Data Manipulation:** Pandas, NumPy
* **NLP Technique:** `CountVectorizer` (Bag of Words / Tokenization)
* **Machine Learning Model:** `MultinomialNB` (Multinomial Naive Bayes)
* **Architecture:** `Pipeline` (For clean code and sequential data transformation)

## 🚀 Workflow & Implementation
1. **Dataset Integration:** Imported the 'SMS Spam Collection' dataset and mapped the categorical labels (ham/spam) to binary numerical values (0/1).
2. **Text Vectorization:** Handled the unstructured text data by utilizing `CountVectorizer` to tokenize the messages and build a frequency matrix of words.
3. **Model Training:** Built a classification model using `MultinomialNB`, which is highly effective for discrete feature counts like word frequencies.
4. **Model Evaluation:** The baseline model was evaluated on an unseen test set. Performance metrics including Accuracy, Confusion Matrix, and a Classification Report were generated to analyze the model's precision in detecting spam.

## 📊 Key Takeaways
This project successfully establishes a solid **Baseline Model** for text classification tasks, highlighting the efficiency of the Naive Bayes algorithm when paired with a clean Scikit-Learn Pipeline architecture.