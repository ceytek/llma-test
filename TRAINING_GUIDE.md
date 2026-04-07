# Llama 3 8B Fine-Tuning Rehberi

Bu rehber, HRSmart projesinden toplanan eğitim verisiyle Llama 3 8B modelini
CV–İlan eşleştirmesi için fine-tune etme adımlarını anlatır.

---

## 1. Ön Koşullar

### Donanım
- GPU: En az 16GB VRAM (RTX 3090, RTX 4060 Ti 16GB, RTX 4090, vb.)
- RAM: 32GB+
- Disk: 20GB boş alan

### Yazılım
- Python 3.10+
- CUDA 11.8 veya 12.1 (nvidia-smi ile kontrol et)
- Ollama kurulu (modeli dışa aktarmak için)

---

## 2. Eğitim Verisini Hazırlama

### 2.1 Veriyi sunucudan al

```bash
# Sunucudan local makineye kopyala
scp root@SUNUCU_IP:/opt/hrsmart/AI-Service/training_data/job_matching.jsonl ./data/

# Veya sunucuda önce export et, sonra kopyala
ssh root@SUNUCU_IP "bash /opt/hrsmart/AI-Service/export_training_data.sh"
scp root@SUNUCU_IP:/opt/hrsmart/AI-Service/training_data/*.jsonl ./data/
```

### 2.2 Veri formatı

Her satır (JSONL) şu formatta — bu zaten ChatML formatı, fine-tuning araçları doğrudan okuyabilir:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert HR analyst..."},
    {"role": "user", "content": "Analyze this candidate against the job..."},
    {"role": "assistant", "content": "{\"overall_score\": 34, \"recommendation\": \"not_recommended\", ...}"}
  ],
  "_meta": {
    "task": "job_matching",
    "ts": "2026-04-02T09:59:06.021555+00:00",
    "job_title": "Human Resources Manager",
    "candidate_name": "ULAŞ SERTBAŞ",
    "overall_score": 34
  }
}
```

### 2.3 Veriyi temizle (opsiyonel ama önerilir)

`_meta` alanını eğitimden çıkar, sadece `messages` kalsın:

```python
import json

with open("data/job_matching.jsonl", "r") as f_in, \
     open("data/train.jsonl", "w") as f_out:
    for line in f_in:
        record = json.loads(line)
        clean = {"messages": record["messages"]}
        f_out.write(json.dumps(clean, ensure_ascii=False) + "\n")

print("Temizlenmiş veri: data/train.jsonl")
```

### 2.4 Train/validation split

```python
import json, random

with open("data/train.jsonl", "r") as f:
    lines = f.readlines()

random.shuffle(lines)
split = int(len(lines) * 0.9)  # %90 train, %10 val

with open("data/train_split.jsonl", "w") as f:
    f.writelines(lines[:split])

with open("data/val_split.jsonl", "w") as f:
    f.writelines(lines[split:])

print(f"Train: {split} kayıt, Val: {len(lines) - split} kayıt")
```

---

## 3. Fine-Tuning (Unsloth ile — En Kolay Yol)

### 3.1 Kurulum

```bash
# Yeni bir ortam oluştur
conda create -n llama-ft python=3.11 -y
conda activate llama-ft

# Unsloth kur (otomatik olarak PyTorch + CUDA yükler)
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
```

### 3.2 Eğitim scripti

`fine_tune.py` dosyası oluştur:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ── 1. Modeli yükle (4-bit quantized — 16GB VRAM'e sığar) ────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# ── 2. LoRA adaptörlerini ekle ────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                  # LoRA rank — 16 iyi bir başlangıç
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # bellek tasarrufu
)

# ── 3. Veriyi yükle ──────────────────────────────────────────────
dataset = load_dataset("json", data_files={
    "train": "data/train_split.jsonl",
    "validation": "data/val_split.jsonl",
})

# ChatML formatına dönüştür
def format_chat(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = dataset.map(format_chat)

# ── 4. Eğitim ayarları ───────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,              # 200-500 örnekle 3 epoch yeterli
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
    ),
)

# ── 5. Eğit! ─────────────────────────────────────────────────────
trainer.train()

# ── 6. Kaydet ─────────────────────────────────────────────────────
model.save_pretrained("./hrsmart-llama3-lora")
tokenizer.save_pretrained("./hrsmart-llama3-lora")
print("Eğitim tamamlandı! LoRA adaptörü: ./hrsmart-llama3-lora")
```

### 3.3 Eğitimi başlat

```bash
python fine_tune.py
```

Beklenen süre: 200-500 örnekle **30-90 dakika** (tek GPU).

---

## 4. Eğitilmiş Modeli Ollama'ya Yükleme

### 4.1 LoRA'yı base model ile birleştir (merge)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./hrsmart-llama3-lora",
    max_seq_length=4096,
    load_in_4bit=False,  # merge için full precision
)

# GGUF formatına dönüştür (Ollama'nın kullandığı format)
model.save_pretrained_gguf(
    "hrsmart-llama3-gguf",
    tokenizer,
    quantization_method="q4_k_m",  # iyi kalite/boyut dengesi
)
print("GGUF dosyası oluşturuldu: hrsmart-llama3-gguf/")
```

### 4.2 Ollama'ya kaydet

`Modelfile` oluştur:

```
FROM ./hrsmart-llama3-gguf/unsloth.Q4_K_M.gguf

PARAMETER temperature 0.1
PARAMETER num_ctx 4096

SYSTEM """You are a strict, objective HR analyst. You score candidates ONLY based on concrete evidence from their CV. Return ONLY valid JSON."""
```

Ollama'ya yükle:

```bash
ollama create hrsmart-matcher -f Modelfile
```

### 4.3 Test et

```bash
ollama run hrsmart-matcher
```

Veya POC uygulamasında sidebar'dan `hrsmart-matcher` modelini seçin.

---

## 5. Özet: Adım Adım

| Adım | Ne Yapılacak | Süre |
|------|-------------|------|
| 1 | Sistemde veri biriktir (200-500 kayıt) | 1-4 hafta (kullanıma bağlı) |
| 2 | Veriyi sunucudan indir ve temizle | 5 dakika |
| 3 | Unsloth kur, fine_tune.py çalıştır | 30-90 dakika |
| 4 | GGUF'a dönüştür, Ollama'ya yükle | 10 dakika |
| 5 | POC ile karşılaştırmalı test yap | 30 dakika |

---

## 6. İpuçları

- **Veri kalitesi > Veri miktarı**: 200 temiz örnek, 1000 gürültülü örnekten iyidir
- **Çeşitlilik önemli**: Farklı ilanlar (yazılım, IK, muhasebe, vb.) ve farklı uyum seviyeleri (düşük, orta, yüksek puan) olsun
- **Epoch sayısı**: 200-500 örnekle 3 epoch yeterli, daha fazlası overfitting yapabilir
- **Değerlendirme**: Fine-tune sonrası aynı test setini hem GPT-4o-mini'ye hem fine-tuned modele gönderip yan yana karşılaştırın
- **İteratif süreç**: İlk sonuçlar yetersizse, daha fazla veri toplayıp tekrar eğitin
