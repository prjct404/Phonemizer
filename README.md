# 🗣️ Persian Text → Finglish → Phoneme → LLM → Speech  
### GE2PE + OpenRouter + Edge-TTS + Prometheus + Grafana

This project converts **Persian text** into **natural speech** using the following pipeline:

  GE2PE for Persian → Finglish phoneme conversion

  LLM (via OpenRouter) to refine / reconstruct phrasing
  
  Microsoft Edge TTS for high-quality neural speech
  
  Prometheus + Grafana for Observability & Monitoring
  
---

## ✨ Features

| Feature | Description |
|--------|-------------|
| ✅ Persian language validation | Rejects non-Persian input |
| ✅ GE2PE phoneme transformation | Converts text → Finglish phonemes |
| ✅ LLM transformation | Uses OpenRouter models (Gemini, Qwen, Claude, etc.) |
| ✅ Neural TTS | Uses `edge-tts` Persian voices |
| ✅ Metrics Export | Pipeline latency, errors, TTS usage, token usage |
| ✅ Grafana Dashboard | Real-time pipeline visibility |


## 🧱 Project Structure

---
```bash
project/
├─ app/
│  ├─ app.py          # Streamlit UI & pipeline flow
│  ├─ metrics.py      # Prometheus counters & histograms
│  ├─ utils.py        # Persian validation, phoneme cleanup
│  ├─ services.py     # GE2PE, OpenRouter, TTS functions
│  └─ config.py       # API keys & constants
├─ monitoring/
│  └─ prometheus.yml  # Prometheus scrape configuration
└─ docker-compose.yml # Prometheus + Grafana stack
 ``` 

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
 ```  
2. **run**:  
 ```bash
   cp env.example .env
   docker compose up -d
   streamlit run main.py
   ```  
