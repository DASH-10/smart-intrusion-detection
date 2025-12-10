Nonlinear diffusion filtering (Perona–Malik) mini-proje

Bu proje, Perona–Malik tabanli (PM1, PM2, Charbonnier) nonlineer difuzyonla kenar koruyucu gurultu giderme yapiyor; gri ve renkli versiyonlar var. Linear difuzyon (isi denklemi) karsilastirma amacli eklenmistir.

Klasor yapisi
- code/
  - nonlinear_diffusion.py    ana script, siniflar ve CLI menusu
  - diffusivity_functions.py  PM1, PM2, Charbonnier g(|grad|) fonksiyonlari
  - analysis.py               karsilastirma ve istatistik grafik fonksiyonlari
  - utils.py                  I/O, metrikler (PSNR, SSIM), sentetik goruntu
- html/
  - index.html                kisa rapor, <img> yer tutuculari
  - style.css                 basit stil
  - results/, plots/          scriptin kaydedecegi cikti klasorleri
- data/ (istege bagli)        kendi test goruntunu buraya koyabilirsin (test_image.png)

Gereksinimler
- Python 3.9+
- Paketler: numpy, opencv-python, matplotlib, scipy
  Kurulum: py -m pip install --upgrade numpy opencv-python matplotlib scipy

Calistirma (Windows PowerShell icin)
1) Proje klasorune gir:
   cd C:\Users\pc\OneDrive\Documents\my_projects\Omar-Issa-PA4
2) Scripti baslat:
   py code\nonlinear_diffusion.py
3) Menuden sec:
   1) Gri ton (linear vs PM1/PM2/Charbonnier)
   2) Renkli (PM1)
   3) Parametre taramasi (lambda ve sigma)

Girdi goruntu
- Oncelik sirasi: data/test_image.png, sonra proje kokundeki test_image.png.
- Bu dosyalar yoksa otomatik sentetik goruntu (sekiller + gurultu) uretir ve devam eder.

Kayitli ciktilar
- html/results/: ornek cikti goruntuleri (karsilastirma, renkli difuzyon, vb.)
- html/plots/: istatistik ve parametre taramasi grafiklari
- Rapor: html/index.html dosyasini tarayicida ac; script sonrasi olusan PNG'leri gosterir.

Notlar
- PSNR ve SSIM metrikleri konsola yazdirilir (uygunsa).
- dt ve iterasyon sayisi sabit zaman adimli acik Euler; buyuk dt/iterasyon icin kararliliga dikkat.


py Omar-Issa-PA4\code\nonlinear_diffusion.py
