
import numpy as np
import pandas as pd
import matplotlib


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

from sklearn.tree import (DecisionTreeClassifier, export_text,
                          plot_tree)
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

print("=" * 60)
print("  KARAR AĞACI — KREDİ DEĞERLENDİRME SİSTEMİ")
print("=" * 60)

# ── 2. SENTETİK VERİ OLUŞTURMA ───────────────────────────────────
print("\n[1/6] Veri seti oluşturuluyor...")

np.random.seed(42)
N = 1_000  # toplam başvuru sayısı

# Demografik özellikler
yas = np.random.randint(22, 65, N)
calisma_yili = np.clip(np.random.normal(8, 5, N), 0, 40).astype(int)
egitim_kodu = np.random.choice([0, 1, 2, 3], N,
                               p=[0.15, 0.35, 0.35, 0.15])
egitim_map = {0: 'İlköğretim', 1: 'Lise',
              2: 'Lisans', 3: 'Yüksek Lisans / Doktora'}

# Finansal özellikler
yillik_gelir = np.clip(
    np.random.lognormal(10.8, 0.45, N), 18_000, 250_000
).astype(int)
borc_gelir_oran = np.clip(np.random.beta(2, 5, N), 0.02, 0.90)
kredi_skoru = np.clip(
    np.random.normal(660, 80, N), 300, 850
).astype(int)
mulk_sahibi = np.random.choice([0, 1], N, p=[0.45, 0.55])
basvuru_miktari = np.random.randint(5_000, 150_000, N)


# Hedef değişken — iş kurallarına dayalı gerçekçi label
def kredi_karari(yas, gelir, skor, borc, egitim, mulk):
    puan = 0
    puan += (skor - 500) / 350 * 40
    puan += min(gelir / 10_000, 15)
    puan -= borc * 25
    puan += mulk * 8
    puan += egitim * 3
    puan += (min(yas, 50) - 22) * 0.2
    puan += np.random.normal(0, 4)
    return int(puan > 30)


y = np.array([
    kredi_karari(yas[i], yillik_gelir[i], kredi_skoru[i],
                 borc_gelir_oran[i], egitim_kodu[i], mulk_sahibi[i])
    for i in range(N)
])

df = pd.DataFrame({
    'Yas': yas,
    'Calisma_Yili': calisma_yili,
    'Egitim_Kodu': egitim_kodu,
    'Yillik_Gelir': yillik_gelir,
    'Borc_Gelir_Orani': borc_gelir_oran.round(3),
    'Kredi_Skoru': kredi_skoru,
    'Mulk_Sahibi': mulk_sahibi,
    'Basvuru_Miktari': basvuru_miktari,
    'Onaylandi': y
})

onay_orani = df['Onaylandi'].mean()
print(f"   ✓ {N} başvuru oluşturuldu.")
print(f"   ✓ Onay oranı: %{onay_orani * 100:.1f}")
print(f"   ✓ Özellik sayısı: {df.shape[1] - 1}")

# ── 3. VERİ ÖN İŞLEME ────────────────────────────────────────────
print("\n[2/6] Veri ön işleme yapılıyor...")

FEATURES = ['Yas', 'Calisma_Yili', 'Egitim_Kodu',
            'Yillik_Gelir', 'Borc_Gelir_Orani',
            'Kredi_Skoru', 'Mulk_Sahibi', 'Basvuru_Miktari']

X = df[FEATURES]
y = df['Onaylandi']

# Özellik mühendisliği: ek türetilmiş değişkenler
X = X.copy()
X['Gelir_Skor_Oran'] = X['Kredi_Skoru'] / (X['Yillik_Gelir'] / 10_000)
X['Yas_Calisma'] = X['Yas'] * X['Calisma_Yili']
X['Net_Yuk_Orani'] = X['Basvuru_Miktari'] / X['Yillik_Gelir']
FEATURES_EXT = FEATURES + ['Gelir_Skor_Oran', 'Yas_Calisma', 'Net_Yuk_Orani']

X_train, X_test, y_train, y_test = train_test_split(
    X[FEATURES_EXT], y, test_size=0.20,
    random_state=42, stratify=y
)

print(f"   ✓ Eğitim seti: {len(X_train)} örnek")
print(f"   ✓ Test seti  : {len(X_test)} örnek")
print(f"   ✓ Türetilmiş özellikler eklendi (+3)")

# ── 4. HİPERPARAMETRE OPTİMİZASYONU ─────────────────────────────
print("\n[3/6] GridSearchCV ile hiperparametre optimizasyonu...")

param_grid = {
    'max_depth': [3, 4, 5, 6, 8, None],
    'min_samples_split': [10, 20, 40, 80],
    'min_samples_leaf': [5, 10, 20],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0.0, 0.001, 0.005]
}

base_model = DecisionTreeClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"   ✓ En iyi parametreler: {best_params}")
print(f"   ✓ Cross-val AUC     : {grid_search.best_score_:.4f}")

# ── 5. FINAL MODEL EĞİTİMİ ───────────────────────────────────────
print("\n[4/6] Final model eğitiliyor...")

model = grid_search.best_estimator_

# Test seti değerlendirmesi
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f"\n   {'─' * 45}")
print(f"   TEST SETİ SONUÇLARI")
print(f"   {'─' * 45}")
print(classification_report(y_test, y_pred,
                            target_names=['Reddedildi', 'Onaylandı'],
                            digits=4))
print(f"   ROC-AUC Skoru: {auc:.4f}")

# Cross-validation sonuçları
cv_scores = cross_val_score(model, X[FEATURES_EXT], y,
                            cv=cv, scoring='roc_auc')
print(f"\n   5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ── 6. GÖRSELLEŞTİRME ────────────────────────────────────────────
print("\n[5/6] Görseller oluşturuluyor...")

PALETTE = {
    'bg': '#0f1117',
    'surface': '#1a1d27',
    'card': '#242736',
    'accent': '#7f77dd',
    'green': '#1d9e75',
    'red': '#e24b4a',
    'amber': '#ef9f27',
    'text': '#e8e6de',
    'muted': '#888780',
}


fig = plt.figure(figsize=(20, 16), facecolor=PALETTE['bg'])
fig.suptitle('Karar Ağacı — Kredi Değerlendirme Sistemi',
             fontsize=18, fontweight='bold',
             color=PALETTE['text'], y=0.97)

gs = GridSpec(3, 3, figure=fig,
              hspace=0.40, wspace=0.35,
              top=0.93, bottom=0.04,
              left=0.06, right=0.97)

# 1. KARAR AĞACI ÇİZİMİ (Üstteki iki satırı ve üç sütunu kaplasın)
ax_tree = fig.add_subplot(gs[0:2, :])
ax_tree.set_facecolor(PALETTE['surface'])
ax_tree.set_title("Eğitilmiş Karar Ağacı Yapısı", color=PALETTE['text'], fontsize=14)

plot_tree(model,
          feature_names=FEATURES_EXT,
          class_names=['Red', 'Onay'],
          filled=True,
          rounded=True,
          ax=ax_tree,
          fontsize=9)

# 2. KARMAŞIKLIK MATRİSİ (Alt sol köşe)
ax_cm = fig.add_subplot(gs[2, 0])
ax_cm.set_facecolor(PALETTE['surface'])
ax_cm.set_title("Karmaşıklık Matrisi (Test Seti)", color=PALETTE['text'])

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Red', 'Onay'])
disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)

# Grafiklerin arka plan renklerini düzeltme
for text in ax_cm.texts: text.set_color("black")
for spine in ax_cm.spines.values(): spine.set_color(PALETTE['muted'])
ax_cm.tick_params(colors=PALETTE['text'])
ax_cm.xaxis.label.set_color(PALETTE['text'])
ax_cm.yaxis.label.set_color(PALETTE['text'])

print("[6/6] İşlem tamamlandı. Grafik ekrana yansıtılıyor...")
plt.show()




