# POC: Llama 3 ile CV–İlan Eşleştirme

Local LLM (Llama 3 8B) kullanarak CV analizi ve iş ilanı eşleştirmesi yapan bağımsız POC.
Hiçbir veri cloud'a gönderilmez — tamamı local makinenizde çalışır.

## Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.ai) kurulu ve çalışır durumda
- Llama 3 modeli indirilmiş

## Kurulum (Local Makinenizde)

```bash
# 1. Ollama'nın çalıştığından emin olun
ollama serve

# 2. Model yoksa indirin
ollama pull llama3:8b

# 3. Bu klasöre gelin
cd poc-llama

# 4. Virtual environment (önerilir)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 5. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 6. Uygulamayı başlatın
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` açılacak.

## Kullanım

1. Sol panelden **ilan bilgilerini** doldurun (pozisyon, açıklama, aranan yetenekler)
2. Sağ panelden **CV yükleyin** (PDF veya DOCX)
3. **Analiz Et** butonuna basın
4. Sonuçları inceleyin: genel puan, puan dağılımı, güçlü/zayıf yanlar

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `app.py` | Streamlit UI — ana uygulama |
| `llama_client.py` | Ollama REST API istemcisi |
| `prompts.py` | CV parse ve eşleştirme prompt'ları |
| `cv_extractor.py` | PDF/DOCX metin çıkarma |
| `requirements.txt` | Python bağımlılıkları |

## Notlar

- Llama 3 8B, GPT-4o-mini'ye kıyasla daha yavaş yanıt verebilir (özellikle CPU'da)
- Türkçe CV'lerde doğruluk farkı olabilir — bu POC'nin amacı tam da bunu test etmek
- Sidebar'dan model ve temperature ayarlarını değiştirebilirsiniz
