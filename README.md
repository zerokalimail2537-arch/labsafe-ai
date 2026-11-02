# LabSafe AI — Render Free

Python + Gradio + Gemini AI สำหรับตรวจสอบ PPE จากภาพ โดยผู้ใช้งานกรอก API Key เอง

## Render settings
- Service: **Web Service** (not Static Site)
- Language: **Python 3**
- Branch: `main`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`
- Instance Type: **Free**
- Environment Variables: optional `GEMINI_MODEL=gemini-2.5-flash`

No server-side Gemini API key is required.

## Local run
```bash
python -m pip install -r requirements.txt
python app.py
```
Open http://localhost:10000

## Privacy
API keys are sent to the Render-hosted app and then to Google's Gemini API. Do not enable request logging or third-party monitoring of request bodies. Uploaded images are temporarily processed on the server and may exist in Gradio's temporary cache until cleanup. The app does not intentionally persist API keys, images or analysis results in a database. Do not submit confidential images without appropriate permission.
