
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
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
    puan += (skor - 500) / 350 * 40  # kredi skoru ağırlığı
    puan += min(gelir / 10_000, 15)  # gelir katkısı
    puan -= borc * 25  # borç oranı cezası
    puan += mulk * 8  # mülk sahipliği bonusu
    puan += egitim * 3  # eğitim bonusu
    puan += (min(yas, 50) - 22) * 0.2  # deneyim bonusu
    puan += np.random.normal(0, 4)  # gürültü (gerçek hayat!)
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
    n_jobs=-1,
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

# ── 6a. Ağaç Yapısı (derinlik=3 özet) ─────────────────────────────
ax_tree = fig.add_subplot(gs[0, :])
ax_tree.set_facecolor(PALETTE['surface'])
ax_tree.set_title('Karar Ağacı Yapısı (Derinlik ≤ 4)',
                  color=PALETTE['text'], pad=8, fontsize=11)

viz_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20,
    criterion=best_params['criterion'],
    random_state=42
)
viz_model.fit(X_train, y_train)

plot_tree(
    viz_model,
    feature_names=FEATURES_EXT,
    class_names=['Reddedildi', 'Onaylandı'],
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax_tree,
    impurity=True,
    proportion=False,
    node_ids=False,
)
for text in ax_tree.texts:
    text.set_color('#0a0a0a')

# ── 6b. Özellik Önem Dereceleri ────────────────────────────────────
ax_imp = fig.add_subplot(gs[1, 0])
ax_imp.set_facecolor(PALETTE['card'])

importances = pd.Series(model.feature_importances_,
                        index=FEATURES_EXT).sort_values()
colors_bar = [PALETTE['accent'] if v >= importances.quantile(0.75)
              else PALETTE['muted'] for v in importances]

bars = ax_imp.barh(importances.index, importances.values,
                   color=colors_bar, height=0.6)
ax_imp.set_title('Özellik Önem Dereceleri',
                 color=PALETTE['text'], pad=6, fontsize=10)
ax_imp.set_xlabel('Gini Importance', color=PALETTE['muted'], fontsize=8)
ax_imp.tick_params(colors=PALETTE['muted'], labelsize=8)
ax_imp.spines['top'].set_visible(False)
ax_imp.spines['right'].set_visible(False)
for sp in ['bottom', 'left']:
    ax_imp.spines[sp].set_color(PALETTE['muted'])
ax_imp.set_facecolor(PALETTE['card'])
for bar, val in zip(bars, importances.values):
    ax_imp.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center',
                fontsize=7, color=PALETTE['text'])

# ── 6c. ROC Eğrisi ─────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 1])
ax_roc.set_facecolor(PALETTE['card'])

fpr, tpr, _ = roc_curve(y_test, y_proba)
ax_roc.plot(fpr, tpr, color=PALETTE['accent'], lw=2,
            label=f'AUC = {auc:.3f}')
ax_roc.plot([0, 1], [0, 1], '--', color=PALETTE['muted'],
            lw=1, label='Rastgele Sınıflandırıcı')
ax_roc.fill_between(fpr, tpr, alpha=0.1, color=PALETTE['accent'])
ax_roc.set_title('ROC Eğrisi', color=PALETTE['text'], pad=6, fontsize=10)
ax_roc.set_xlabel('Yanlış Pozitif Oranı', color=PALETTE['muted'], fontsize=8)
ax_roc.set_ylabel('Doğru Pozitif Oranı', color=PALETTE['muted'], fontsize=8)
ax_roc.tick_params(colors=PALETTE['muted'], labelsize=8)
ax_roc.legend(fontsize=8, facecolor=PALETTE['surface'],
              labelcolor=PALETTE['text'])
for sp in ax_roc.spines.values():
    sp.set_color(PALETTE['muted'])
ax_roc.set_facecolor(PALETTE['card'])

# ── 6d. Karışıklık Matrisi ─────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1, 2])
ax_cm.set_facecolor(PALETTE['card'])

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm,
                                    display_labels=['Reddedildi', 'Onaylandı'])
cm_display.plot(ax=ax_cm, colorbar=False,
                cmap='PuBu', values_format='d')
ax_cm.set_title('Karışıklık Matrisi', color=PALETTE['text'],
                pad=6, fontsize=10)
ax_cm.tick_params(colors=PALETTE['muted'], labelsize=8)
ax_cm.set_xlabel('Tahmin', color=PALETTE['muted'], fontsize=8)
ax_cm.set_ylabel('Gerçek', color=PALETTE['muted'], fontsize=8)
ax_cm.set_facecolor(PALETTE['card'])

# ── 6e. Derinlik vs Performans ─────────────────────────────────────
ax_depth = fig.add_subplot(gs[2, 0])
ax_depth.set_facecolor(PALETTE['card'])

depths = list(range(1, 16))
train_s, test_s = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_s.append(roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]))
    test_s.append(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))

ax_depth.plot(depths, train_s, '-o', color=PALETTE['amber'],
              ms=4, lw=1.5, label='Eğitim AUC')
ax_depth.plot(depths, test_s, '-o', color=PALETTE['green'],
              ms=4, lw=1.5, label='Test AUC')
best_d = depths[np.argmax(test_s)]
ax_depth.axvline(best_d, color=PALETTE['accent'],
                 lw=1.2, ls='--', alpha=0.7,
                 label=f'Optimum: {best_d}')
ax_depth.fill_between(depths, train_s, test_s,
                      alpha=0.12, color=PALETTE['red'],
                      label='Overfitting bölgesi')
ax_depth.set_title('Derinlik vs. AUC (Overfitting Analizi)',
                   color=PALETTE['text'], pad=6, fontsize=10)
ax_depth.set_xlabel('Ağaç Derinliği', color=PALETTE['muted'], fontsize=8)
ax_depth.set_ylabel('ROC-AUC', color=PALETTE['muted'], fontsize=8)
ax_depth.tick_params(colors=PALETTE['muted'], labelsize=8)
ax_depth.legend(fontsize=8, facecolor=PALETTE['surface'],
                labelcolor=PALETTE['text'])
for sp in ax_depth.spines.values():
    sp.set_color(PALETTE['muted'])
ax_depth.set_facecolor(PALETTE['card'])

# ── 6f. Kredi Skoru vs Onay Kararı ────────────────────────────────
ax_scat = fig.add_subplot(gs[2, 1])
ax_scat.set_facecolor(PALETTE['card'])

mask_on = y_test == 1
mask_re = y_test == 0
ax_scat.scatter(
    X_test.loc[mask_re, 'Kredi_Skoru'],
    X_test.loc[mask_re, 'Borc_Gelir_Orani'],
    c=PALETTE['red'], alpha=0.5, s=18, label='Reddedildi'
)
ax_scat.scatter(
    X_test.loc[mask_on, 'Kredi_Skoru'],
    X_test.loc[mask_on, 'Borc_Gelir_Orani'],
    c=PALETTE['green'], alpha=0.5, s=18, label='Onaylandı'
)
ax_scat.set_title('Karar Sınırı: Kredi Skoru vs Borç Oranı',
                  color=PALETTE['text'], pad=6, fontsize=10)
ax_scat.set_xlabel('Kredi Skoru', color=PALETTE['muted'], fontsize=8)
ax_scat.set_ylabel('Borç / Gelir Oranı', color=PALETTE['muted'], fontsize=8)
ax_scat.tick_params(colors=PALETTE['muted'], labelsize=8)
ax_scat.legend(fontsize=8, facecolor=PALETTE['surface'],
               labelcolor=PALETTE['text'])
for sp in ax_scat.spines.values():
    sp.set_color(PALETTE['muted'])
ax_scat.set_facecolor(PALETTE['card'])

# ── 6g. Özet Metrikler Panel ──────────────────────────────────────
ax_sum = fig.add_subplot(gs[2, 2])
ax_sum.set_facecolor(PALETTE['card'])
ax_sum.axis('off')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
depth_val = model.get_depth()
leaves_val = model.get_n_leaves()

metrics = [
    ('ROC-AUC', f'{auc:.4f}', PALETTE['accent']),
    ('Accuracy', f'{acc:.4f}', PALETTE['green']),
    ('F1-Score', f'{f1:.4f}', PALETTE['amber']),
    ('Precision', f'{prec:.4f}', PALETTE['muted']),
    ('Recall', f'{rec:.4f}', PALETTE['muted']),
    ('Ağaç Derinliği', str(depth_val), PALETTE['muted']),
    ('Yaprak Sayısı', str(leaves_val), PALETTE['muted']),
    ('CV AUC', f'{cv_scores.mean():.4f} ± {cv_scores.std():.4f}',
     PALETTE['accent']),
]

ax_sum.set_title('Model Özet Metrikleri',
                 color=PALETTE['text'], pad=6, fontsize=10)
for i, (label, value, color) in enumerate(metrics):
    y_pos = 0.95 - i * 0.115
    ax_sum.text(0.05, y_pos, label + ':',
                transform=ax_sum.transAxes,
                fontsize=9, color=PALETTE['muted'], va='top')
    ax_sum.text(0.55, y_pos, value,
                transform=ax_sum.transAxes,
                fontsize=9, fontweight='bold',
                color=color, va='top')

plt.show()
plt.close()
print("   ✓ Görsel kaydedildi → decision_tree_analiz.png")

# ── 7. CANLI TAHMİN FONKSİYONU ───────────────────────────────────
print("\n[6/6] Canlı tahmin demonstrasyonu...")


def tahmin_et(yas, calisma_yili, egitim_kodu,
              yillik_gelir, borc_gelir_oran,
              kredi_skoru, mulk_sahibi, basvuru_miktari):
    """
    Yeni bir başvuruyu değerlendirir.

    Parametreler
    ─────────────
    yas              : int   — başvuranın yaşı
    calisma_yili     : int   — toplam çalışma yılı
    egitim_kodu      : int   — 0=İlk., 1=Lise, 2=Lisans, 3=Y.Lis./Dok.
    yillik_gelir     : int   — TL cinsinden yıllık net gelir
    borc_gelir_oran  : float — 0-1 arasında mevcut borç/gelir oranı
    kredi_skoru      : int   — 300-850 arasında kredi notu
    mulk_sahibi      : int   — 0=Hayır, 1=Evet (ev/araba)
    basvuru_miktari  : int   — talep edilen kredi miktarı (TL)

    Dönüş
    ───────
    dict — karar ve olasılık bilgileri
    """
    row = pd.DataFrame([{
        'Yas': yas,
        'Calisma_Yili': calisma_yili,
        'Egitim_Kodu': egitim_kodu,
        'Yillik_Gelir': yillik_gelir,
        'Borc_Gelir_Orani': borc_gelir_oran,
        'Kredi_Skoru': kredi_skoru,
        'Mulk_Sahibi': mulk_sahibi,
        'Basvuru_Miktari': basvuru_miktari,
        # türetilmiş özellikler
        'Gelir_Skor_Oran': kredi_skoru / (yillik_gelir / 10_000),
        'Yas_Calisma': yas * calisma_yili,
        'Net_Yuk_Orani': basvuru_miktari / yillik_gelir,
    }])

    karar = model.predict(row)[0]
    olasilik = model.predict_proba(row)[0]

    # Karar yolu (hangi kurallara girdi?)
    decision_path = model.decision_path(row)
    node_ids = decision_path.indices

    return {
        'karar': 'ONAYLANDI ✓' if karar == 1 else 'REDDEDİLDİ ✗',
        'onay_olasiligi': f'%{olasilik[1] * 100:.1f}',
        'red_olasiligi': f'%{olasilik[0] * 100:.1f}',
        'karar_kodu': karar,
        'gecilen_dugum': len(node_ids),
    }


# Örnek başvurular
ornekler = [
    {
        'ad': 'Ahmet K. — Güçlü Profil',
        'yas': 38, 'calisma_yili': 14, 'egitim_kodu': 2,
        'yillik_gelir': 95_000, 'borc_gelir_oran': 0.18,
        'kredi_skoru': 730, 'mulk_sahibi': 1,
        'basvuru_miktari': 40_000,
    },
    {
        'ad': 'Elif M. — Orta Profil',
        'yas': 29, 'calisma_yili': 5, 'egitim_kodu': 2,
        'yillik_gelir': 48_000, 'borc_gelir_oran': 0.42,
        'kredi_skoru': 610, 'mulk_sahibi': 0,
        'basvuru_miktari': 25_000,
    },
    {
        'ad': 'Murat T. — Riskli Profil',
        'yas': 24, 'calisma_yili': 2, 'egitim_kodu': 1,
        'yillik_gelir': 28_000, 'borc_gelir_oran': 0.68,
        'kredi_skoru': 520, 'mulk_sahibi': 0,
        'basvuru_miktari': 35_000,
    },
]

print(f"\n   {'─' * 50}")
print("   ÖRNEK BAŞVURU TAHMİNLERİ")
print(f"   {'─' * 50}")
for o in ornekler:
    sonuc = tahmin_et(
        o['yas'], o['calisma_yili'], o['egitim_kodu'],
        o['yillik_gelir'], o['borc_gelir_oran'],
        o['kredi_skoru'], o['mulk_sahibi'], o['basvuru_miktari']
    )
    print(f"\n   Başvuran : {o['ad']}")
    print(f"   Karar    : {sonuc['karar']}")
    print(f"   Onay Olas.: {sonuc['onay_olasiligi']}")
    print(f"   Red Olas. : {sonuc['red_olasiligi']}")
    print(f"   Düğüm Seyahati: {sonuc['gecilen_dugum']} düğüm")

# ── 8. KURAL ÇIKARIMI (İnsanların Anlayacağı Format) ─────────────
print(f"\n{'=' * 60}")
print("  AĞAÇTAN ÇIKARILAN KURALLAR (İlk 15 Satır)")
print(f"{'=' * 60}")
tree_rules = export_text(viz_model,
                         feature_names=FEATURES_EXT,
                         max_depth=3)
for line in tree_rules.split('\n')[:18]:
    print(' ', line)

print(f"\n{'=' * 60}")
print("  TÜZÜM: decision_tree_analiz.png görsel dosyaya kaydedildi.")
print(f"{'=' * 60}\n")
