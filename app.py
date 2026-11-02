[สรุปรายการที่พบ ไม่พบ และไม่สามารถยืนยันได้]

หมายเหตุ: การวิเคราะห์จากภาพเป็นการประเมินเบื้องต้น ไม่ใช่การรับรองความปลอดภัยในการเข้าห้องปฏิบัติการ และไม่สามารถรับรองมาตรฐานอุปกรณ์จากภาพเพียงอย่างเดียว
"""


def check_object(image: Image.Image, expected_list: list[str], api_key: str) -> str:
    if image is None:
        return "กรุณาอัปโหลดหรือถ่ายภาพก่อนค่ะ"
    if not expected_list:
        return "กรุณาเลือกอุปกรณ์ PPE อย่างน้อย 1 รายการค่ะ"
    api_key = (api_key or "").strip()
    if not api_key:
        return "กรุณากรอก Gemini API Key ก่อนค่ะ"
    selected = [item for item in PPE_ITEMS if item in expected_list]
    if not selected:
        return "ไม่มีรายการ PPE ที่รองรับ"

    try:
        img = image.convert("RGB")
        img.thumbnail((1600, 1600))
        prompt = PROMPT.format(items=", ".join(selected))
        with genai.Client(api_key=api_key) as client:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(temperature=0.1),
                    )
                    answer = (response.text or "").strip()
                    return answer or "AI ไม่ส่งผลการวิเคราะห์กลับมา กรุณาลองภาพอื่นอีกครั้ง"
                except Exception as exc:
                    status = getattr(exc, "code", None)
                    message = str(exc)
                    retryable = status in (429, 500, 502, 503, 504) or any(
                        token in message for token in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
                    )
                    if retryable and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    if status in (400, 401, 403) or "API_KEY_INVALID" in message or "API key not valid" in message:
                        return "ไม่สามารถยืนยัน Gemini API Key หรือไม่มีสิทธิ์ใช้โมเดล กรุณาตรวจสอบ Key และสิทธิ์ใน Google AI Studio"
                    if status == 404 or "NOT_FOUND" in message:
                        return f"ไม่พบโมเดล {MODEL} กรุณาตรวจสอบชื่อโมเดลที่รองรับใน Gemini API"
                    if status == 429 or "RESOURCE_EXHAUSTED" in message:
                        return "Gemini API ถึงขีดจำกัดการใช้งานของ Key นี้ กรุณารอสักครู่หรือตรวจสอบโควตา"
                    logging.warning("Gemini request failed: %s", type(exc).__name__)
                    return "เกิดข้อผิดพลาดในการประมวลผล กรุณาตรวจสอบ API Key, โควตา หรือทดลองใหม่ภายหลัง"
    except Exception:
        logging.exception("Image processing failed")
        return "ไม่สามารถประมวลผลภาพนี้ได้ กรุณาใช้ไฟล์ JPG หรือ PNG ที่ถูกต้อง"


CSS = """
.gradio-container {max-width: 1050px !important; margin: auto !important;}
#intro {text-align:center; padding:12px 0 4px;}
#intro h1 {font-size:2.2rem;}
"""

with gr.Blocks(
    title="LabSafe AI | PPE Checker",
    analytics_enabled=False,
    delete_cache=(3600, 3600),
) as demo:
    gr.Markdown(
        "# 🧪 LabSafe AI\n### ระบบช่วยตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE) จากภาพด้วย Gemini AI\nเลือกอุปกรณ์ อัปโหลดหรือถ่ายภาพ และใช้ Gemini API Key ของคุณเอง",
        elem_id="intro",
    )
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Gemini API Key ของคุณ",
                type="password",
                placeholder="กรอก API Key จาก Google AI Studio",
                info="Key ใช้สำหรับคำขอนี้ ไม่ถูกฝังไว้ในโค้ดหรือเก็บเป็นตัวแปรส่วนกลาง",
            )
            expected = gr.CheckboxGroup(
                choices=PPE_ITEMS, value=PPE_ITEMS, label="เลือกอุปกรณ์ PPE ที่ต้องการตรวจสอบ"
            )
            image = gr.Image(
                type="pil", sources=["upload", "webcam"], label="อัปโหลดภาพหรือถ่ายภาพ",
                image_mode="RGB", height=340,
            )
            button = gr.Button("🔍 ตรวจสอบ PPE", variant="primary")
            clear = gr.Button("ล้างภาพและผลการตรวจสอบ", variant="secondary")
        with gr.Column(scale=1):
            result = gr.Textbox(label="ผลการตรวจสอบ", lines=20, buttons=["copy"])
            gr.Markdown(
                "**หมายเหตุ:** ผลจาก AI เป็นการประเมินเบื้องต้น ไม่ใช่การรับรองความปลอดภัยหรือมาตรฐาน PPE\n\n"
                "สร้าง API Key ได้ที่ [Google AI Studio](https://aistudio.google.com/apikey) กรุณาอย่าใช้ Key ของผู้อื่น"
            )
    button.click(
        fn=check_object,
        inputs=[image, expected, api_key_input],
        outputs=result,
        api_name=False,
        concurrency_limit=2,
    )
    clear.click(
        fn=lambda: (None, ""),
        inputs=None,
        outputs=[image, result],
        api_name=False,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20, api_open=False)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        css=CSS,
        show_error=False,
    )
