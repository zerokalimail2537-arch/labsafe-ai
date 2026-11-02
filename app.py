
import os
import time
import logging

import gradio as gr
from google import genai
from google.genai import types
from PIL import Image

# =====================================
# ตั้งค่าระบบ
# =====================================

logging.basicConfig(level=logging.INFO)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PPE_ITEMS = [
    "ถุงมือ",
    "แว่นตานิรภัย",
    "เสื้อช็อป",
    "สายสะท้อนแสง",
    "หน้ากากอนามัย",
]

# =====================================
# คำสั่งสำหรับ Gemini AI
# =====================================

PROMPT = """
คุณเป็น AI ผู้ช่วยตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE)
สำหรับห้องปฏิบัติการเคมี

กรุณาวิเคราะห์ภาพที่แนบมาอย่างละเอียด

ตรวจสอบเฉพาะอุปกรณ์ต่อไปนี้:
{items}

ข้อกำหนดในการตรวจสอบ:

1. ถุงมือ
- ตรวจสอบถุงมือทุกสี
- รวมถึงถุงมือยางและถุงมือไนไตรล์
- ตรวจสอบว่ามีการสวมถุงมือที่มือจริงหรือไม่
- ห้ามใช้สีเป็นเงื่อนไขตัดสินเพียงอย่างเดียว

2. แว่นตานิรภัย
- ตรวจสอบแว่นครอบตาและแว่นตานิรภัย
- พิจารณารูปทรง เลนส์ และลักษณะของแว่น
- ห้ามสรุปว่าแว่นสายตาทั่วไปเป็นแว่นนิรภัย

3. เสื้อช็อป
- ตรวจสอบเสื้อช็อปวิศวกรรม
- เสื้อเครื่องแบบช่างและเสื้อปฏิบัติงาน
- หากแยกจากเสื้อทั่วไปไม่ได้
  ให้ระบุว่าไม่สามารถยืนยันได้

4. สายสะท้อนแสง
- ตรวจสอบแถบสะท้อนแสงบนเสื้อ
- ไม่จำกัดสี เช่น เงิน เหลือง ส้ม
- แถบสีสดเพียงอย่างเดียวไม่ยืนยัน
  คุณสมบัติการสะท้อนแสง

5. หน้ากากอนามัย
- ตรวจสอบหน้ากากทุกสี
- ตรวจสอบว่าปิดจมูกและปากหรือไม่

หลักการสำคัญ:
- ตรวจสอบทุกรายการที่ผู้ใช้เลือก
- ห้ามเพิ่มรายการที่ไม่ได้เลือก
- วิเคราะห์จากสิ่งที่มองเห็นจริง
- ห้ามแต่งข้อมูลหรือคาดเดา
- หากภาพไม่ชัด ให้ระบุไม่สามารถยืนยันได้
- ไม่สามารถรับรองมาตรฐาน PPE จากภาพเพียงอย่างเดียว

กรุณาตอบเป็นภาษาไทยในรูปแบบ:

ผลการตรวจสอบ PPE

- ชื่ออุปกรณ์: พบ / ไม่พบ / ไม่สามารถยืนยันได้
  รายละเอียด: อธิบายสิ่งที่สังเกตเห็น

สรุปผลการตรวจสอบ:
ระบุรายการที่พบ ไม่พบ และไม่สามารถยืนยันได้

หมายเหตุ:
ผลการวิเคราะห์เป็นเพียงการประเมินเบื้องต้น
ไม่ใช่การรับรองความปลอดภัยในการเข้าห้องปฏิบัติการ
"""

# =====================================
# ฟังก์ชันตรวจสอบ PPE
# =====================================

def check_object(image, expected_list, api_key):

    if image is None:
        return "กรุณาอัปโหลดหรือถ่ายภาพก่อนค่ะ"

    if not expected_list:
        return "กรุณาเลือกอุปกรณ์ PPE อย่างน้อย 1 รายการค่ะ"

    api_key = (api_key or "").strip()

    if not api_key:
        return "กรุณากรอก Gemini API Key ก่อนค่ะ"

    selected_items = [
        item for item in PPE_ITEMS
        if item in expected_list
    ]

    if not selected_items:
        return "ไม่มีรายการ PPE ที่รองรับ"

    try:
        img = image.convert("RGB")
        img.thumbnail((1600, 1600))

        prompt = PROMPT.format(
            items=", ".join(selected_items)
        )

        # สร้าง Client ใหม่สำหรับแต่ละคำขอ
        # ไม่เก็บ API Key เป็น Global Variable

        with genai.Client(api_key=api_key) as client:

            for attempt in range(3):

                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            temperature=0.1
                        ),
                    )

                    answer = (response.text or "").strip()

                    if answer:
                        return answer

                    return (
                        "AI ไม่ส่งผลการวิเคราะห์กลับมา "
                        "กรุณาลองใหม่อีกครั้ง"
                    )

                except Exception as exc:

                    status = getattr(exc, "code", None)
                    message = str(exc)

                    retryable = (
                        status in (429, 500, 502, 503, 504)
                        or any(
                            code in message
                            for code in [
                                "429",
                                "500",
                                "502",
                                "503",
                                "504",
                                "UNAVAILABLE",
                                "RESOURCE_EXHAUSTED",
                            ]
                        )
                    )

                    if retryable and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue

                    if (
                        status in (400, 401, 403)
                        or "API_KEY_INVALID" in message
                        or "API key not valid" in message
                    ):
                        return (
                            "ไม่สามารถใช้งาน Gemini API Key ได้\n"
                            "กรุณาตรวจสอบ API Key และสิทธิ์การใช้งาน"
                        )

                    if status == 404 or "NOT_FOUND" in message:
                        return (
                            f"ไม่พบโมเดล {MODEL}\n"
                            "กรุณาตรวจสอบชื่อโมเดลที่รองรับ"
                        )

                    if (
                        status == 429
                        or "RESOURCE_EXHAUSTED" in message
                    ):
                        return (
                            "Gemini API ถึงขีดจำกัดการใช้งาน\n"
                            "กรุณารอสักครู่แล้วลองใหม่"
                        )

                    logging.warning(
                        "Gemini request failed: %s",
                        type(exc).__name__
                    )

                    return (
                        "เกิดข้อผิดพลาดในการประมวลผล\n"
                        "กรุณาตรวจสอบ API Key หรือทดลองใหม่"
                    )

    except Exception:
        logging.exception("Image processing failed")

        return (
            "ไม่สามารถประมวลผลภาพนี้ได้\n"
            "กรุณาใช้ภาพ JPG หรือ PNG"
        )


# =====================================
# CSS สำหรับตกแต่งหน้าเว็บไซต์
# =====================================

CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}

#intro {
    text-align: center;
    padding: 12px 0 4px;
}

#intro h1 {
    font-size: 2.2rem;
}

#intro h3 {
    font-weight: 400;
}
"""

# =====================================
# สร้างหน้าเว็บไซต์ Gradio
# =====================================

with gr.Blocks(
    title="LabSafe AI | PPE Checker",
    analytics_enabled=False,
    delete_cache=(3600, 3600)
) as demo:

    gr.Markdown(
        """
# 🧪 LabSafe AI

### ระบบตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE)

วิเคราะห์ภาพด้วย Gemini AI

เลือกอุปกรณ์ อัปโหลดภาพ และกรอก Gemini API Key
        """,
        elem_id="intro"
    )

    with gr.Row():

        with gr.Column(scale=1):

            api_key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="กรอก Gemini API Key ของคุณ",
                info="ใช้ API Key ของผู้ใช้งานแต่ละคน"
            )

            expected = gr.CheckboxGroup(
                choices=PPE_ITEMS,
                value=PPE_ITEMS,
                label="เลือกอุปกรณ์ PPE ที่ต้องการตรวจสอบ"
            )

            image = gr.Image(
                type="pil",
                sources=["upload", "webcam"],
                label="อัปโหลดหรือถ่ายภาพ",
                image_mode="RGB",
                height=340
            )

            button = gr.Button(
                "🔍 ตรวจสอบ PPE",
                variant="primary"
            )

            clear = gr.Button(
                "ล้างภาพและผลการตรวจสอบ",
                variant="secondary"
            )

        with gr.Column(scale=1):

            result = gr.Textbox(
                label="ผลการตรวจสอบ",
                lines=20,
                buttons=["copy"]
            )

            gr.Markdown(
                """
**หมายเหตุ**

ผลการวิเคราะห์จาก AI เป็นการประเมินเบื้องต้น
ไม่ใช่การรับรองความปลอดภัยหรือมาตรฐาน PPE

สร้าง Gemini API Key ได้ที่
[Google AI Studio](https://aistudio.google.com/apikey)
                """
            )

    # =====================================
    # ปุ่มตรวจสอบ
    # =====================================

    button.click(
        fn=check_object,
        inputs=[
            image,
            expected,
            api_key_input
        ],
        outputs=result,
        api_name=False,
        concurrency_limit=2
    )

    # =====================================
    # ปุ่มล้างข้อมูล
    # =====================================

    clear.click(
        fn=lambda: (None, ""),
        inputs=[],
        outputs=[image, result],
        api_name=False
    )


# =====================================
# เริ่มต้นเซิร์ฟเวอร์สำหรับ Render
# =====================================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=2,
        max_size=20,
        api_open=False
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        css=CSS,
        show_error=False
    )
