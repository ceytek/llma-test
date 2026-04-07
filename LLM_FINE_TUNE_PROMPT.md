# Bu metni başka bir LLM'e (ChatGPT / Claude) yapıştırın

---

Merhaba. Llama 3 8B modelini kendi özel kullanım alanım için fine-tune etmek istiyorum. Adım adım bana yardımcı ol.

## Mevcut Durumum

- Local makinemde **Ollama** kurulu ve **llama3:8b** modeli çalışıyor.
- GPU'm: [BURAYA GPU MODELINI YAZ — örn: RTX 3090 24GB, RTX 4060 Ti 16GB, vb.]
- OS: [BURAYA OS YAZ — örn: Ubuntu 22.04, Windows 11, macOS]
- Python 3.11 kurulu.

## Ne Yapmak İstiyorum

Bir İK (İnsan Kaynakları) yazılımım var. Bu yazılımda CV'ler ile iş ilanlarını eşleştiren bir AI servisi çalışıyor. Şu an **GPT-4o-mini** kullanıyorum ama verilerin cloud'a gitmemesi ve maliyet avantajı için **local Llama 3 8B** modeline geçmek istiyorum.

GPT-4o-mini'nin ürettiği eğitim verilerini topladım. Bu verilerle Llama 3 8B'yi fine-tune edip GPT-4o-mini'ye yakın kalitede sonuç üretmesini istiyorum.

## Eğitim Verim

Verim **JSONL** formatında, her satır bir kayıt. **ChatML** formatında:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert HR analyst specializing in candidate evaluation and job matching. You provide objective, data-driven assessments of candidate-job fit. IMPORTANT LANGUAGE POLICY: Produce all narrative/free-text content in Turkish (Türkçe)..."
    },
    {
      "role": "user", 
      "content": "You are an expert HR analyst. Analyze the candidate's CV against the job requirements...\n\n**JOB INFORMATION:**\nTitle: Human Resources Manager\nDepartment: İnsan Kaynakları\n...\n\n**CANDIDATE CV INFORMATION:**\nName: ULAŞ SERTBAŞ\n..."
    },
    {
      "role": "assistant",
      "content": "{\"overall_score\": 34, \"recommendation\": \"not_recommended\", \"breakdown\": {\"experience_score\": 10, \"experience_reasoning\": \"Adayın deneyimi...\", ...}, \"matched_skills\": [...], \"missing_skills\": [...], \"strengths\": [...], \"weaknesses\": [...], \"summary\": \"...\"}"
    }
  ]
}
```

- Şu an yaklaşık **14 kayıt** var ama artmaya devam ediyor, hedefim 200-500 kayıt.
- Veriler Türkçe içerik ağırlıklı (ilanlar ve CV'ler Türkçe).
- Assistant çıktısı her zaman yapılandırılmış JSON formatında.

## Eğitim Sonrası Beklentim

Fine-tune edilmiş model şunları yapabilmeli:
1. Türkçe CV'leri ve ilanları doğru anlayabilmeli
2. Tutarlı ve valid JSON formatında çıktı üretmeli
3. Puanlama GPT-4o-mini'ye yakın olmalı (alan uyumsuzluğunda düşük puan, iyi eşleşmede yüksek puan)
4. Doğal Türkçe cümleler kurabilmeli (devrik veya makine çevirisi gibi olmamalı)

## Senden İstediğim

1. **Ortam kurulumu**: Hangi araçları kurmam gerekiyor? (Unsloth, Axolotl, veya başka bir şey öner)
2. **Veri hazırlama**: JSONL dosyamı nasıl temizleyip eğitime hazırlarım?
3. **Eğitim scripti**: QLoRA/LoRA ile fine-tune yapacak tam çalışır Python scripti yaz
4. **Hiperparametre önerileri**: Epoch, learning rate, batch size, LoRA rank, vb. — benim veri boyutum ve GPU'ma göre öner
5. **Eğitim sonrası**: Modeli GGUF formatına çevirip Ollama'ya nasıl yüklerim?
6. **Test ve değerlendirme**: Fine-tuned modeli nasıl test edip GPT-4o-mini ile karşılaştırırım?

Her adımı detaylı anlat, komutları ve kodları ver. Ben yazılım geliştiricisiyim ama ML/fine-tuning konusunda yeniyim.
