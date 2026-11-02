import os
import time
import google.genai as genai
from google.genai import types
import gradio as gr
from PIL import Image

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"


def build_config():
    # ลดเวลา "คิด" ของโมเดล (สาเหตุหลักที่ตอบช้า) และตั้ง timeout ไม่ให้ค้างนาน
    base = dict(temperature=0.1, http_options=types.HttpOptions(timeout=90_000))
    try:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"), **base
        )
    except Exception:
        return types.GenerateContentConfig(**base)


CONFIG = build_config()

def check_object(image, expected_list, api_key):

    if not api_key:
        return "กรุณากรอก Gemini API Key ก่อนค่ะ"

    if image is None:
        return "กรุณาอัปโหลดภาพก่อนค่ะ"

    if not expected_list:
        return "กรุณาเลือกอุปกรณ์ PPE อย่างน้อย 1 รายการค่ะ"

    try:
        img = image.convert("RGB")
        # ย่อภาพก่อนส่ง: ภาพจากมือถือมักใหญ่หลาย MB ทำให้อัปโหลดและประมวลผลช้า
        img.thumbnail((1024, 1024))

        client = genai.Client(api_key=api_key.strip())

        items_to_check = ", ".join(expected_list)

        prompt = f"""
คุณเป็น AI ผู้ช่วยตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE)
สำหรับห้องปฏิบัติการเคมี

กรุณาวิเคราะห์ภาพที่แนบมาอย่างละเอียด
โดยตรวจสอบอุปกรณ์ต่อไปนี้: {items_to_check}

ข้อกำหนดในการตรวจสอบ:

1. ถุงมือ
- ตรวจสอบถุงมือทุกสี เช่น สีขาว น้ำเงิน ม่วง
  เหลือง ดำ เขียว ชมพู และสีอื่น ๆ
- ตรวจสอบทั้งถุงมือยางและถุงมือไนไตรล์
- ห้ามตัดสินว่าถุงมือไม่พบเพียงเพราะสีแตกต่างจากสีขาว

2. แว่นตานิรภัย
- ตรวจสอบแว่นครอบตาและแว่นตานิรภัย
- พิจารณาจากรูปทรง เลนส์ และลักษณะของแว่น

3. เสื้อช็อป
- ตรวจสอบเสื้อช็อปวิศวกรรม เสื้อเครื่องแบบช่าง
  และเสื้อปฏิบัติงาน
- พิจารณาจากรูปแบบเสื้อและลักษณะเครื่องแต่งกาย
- หากไม่สามารถยืนยันได้ว่าเป็นชุดช็อปจริง
  ให้ระบุว่าไม่สามารถยืนยันได้

4. สายสะท้อนแสง
- ตรวจสอบแถบสะท้อนแสงบนเสื้อหรือเครื่องแต่งกาย
- ไม่จำกัดสี เช่น สีเงิน สีเหลือง หรือสีส้ม

5. หน้ากากอนามัย
- ตรวจสอบหน้ากากที่สวมปิดจมูกและปาก
- ตรวจสอบทุกสี เช่น สีขาว ฟ้า เขียว ดำ และสีอื่น ๆ

หลักการสำคัญ:
- ตรวจสอบจากรูปร่าง ลักษณะ และหน้าที่ของอุปกรณ์
  ไม่ใช้สีเป็นเงื่อนไขตัดสินเพียงอย่างเดียว
- ตรวจสอบทุกประเภทที่ผู้ใช้เลือก
- หากพบหลายประเภท ให้รายงานทุกประเภท
- ห้ามแต่งข้อมูลหรือคาดเดาว่ามีอุปกรณ์
  หากมองไม่เห็นชัดเจน
- หากภาพไม่ชัด ให้ระบุว่าไม่สามารถยืนยันได้

กรุณาตอบเป็นภาษาไทยในรูปแบบดังนี้:

ผลการตรวจสอบ PPE

- ชื่ออุปกรณ์: พบ / ไม่พบ / ไม่สามารถยืนยันได้
- รายละเอียด: อธิบายสี ลักษณะ และสิ่งที่สังเกตเห็น

สรุปผลการตรวจสอบ:
อธิบายว่าอุปกรณ์ใดพบและอุปกรณ์ใดไม่พบ

หมายเหตุ:
การวิเคราะห์จากภาพเป็นเพียงการประเมินเบื้องต้น
ไม่ใช่การรับรองความปลอดภัยในการเข้าห้องปฏิบัติการ
"""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[img, prompt],
                    config=CONFIG,
                )
                return (response.text or "").strip() or "AI ไม่ส่งผลการวิเคราะห์กลับมา กรุณาลองใหม่อีกครั้ง"

            except Exception as e:
                error = str(e)

                if "503" in error or "UNAVAILABLE" in error:
                    if attempt < 2:
                        time.sleep(2)
                        continue

                if "429" in error or "RESOURCE_EXHAUSTED" in error:
                    return "ใช้งาน API เกินโควตา กรุณารอสักครู่แล้วลองใหม่อีกครั้ง"

                if "404" in error or "NOT_FOUND" in error:
                    return f"ไม่พบโมเดล {MODEL} สำหรับ API Key นี้ กรุณาตั้งค่า GEMINI_MODEL ใน Render"

                return f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {error}"

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


with gr.Blocks(title="LabSafe AI") as demo:

    gr.Markdown("# 🧪 LabSafe AI")
    gr.Markdown("ระบบช่วยตรวจสอบ PPE จากภาพอัตโนมัติ")

    api_key_input = gr.Textbox(
        label="Gemini API Key",
        type="password",
        placeholder="วาง API Key ที่นี่"
    )

    expected = gr.CheckboxGroup(
        choices=[
            "ถุงมือ",
            "แว่นตานิรภัย",
            "เสื้อช็อป",
            "สายสะท้อนแสง",
            "หน้ากากอนามัย"
        ],
        value=[
            "ถุงมือ",
            "แว่นตานิรภัย",
            "เสื้อช็อป",
            "สายสะท้อนแสง",
            "หน้ากากอนามัย"
        ],
        label="เลือกประเภทวัตถุที่ต้องการตรวจสอบ"
    )

    image = gr.Image(
        type="pil",
        label="ถ่ายรูปหรืออัปโหลดภาพ",
        sources=["upload", "webcam"]
    )

    button = gr.Button(
        "🔍 ตรวจสอบ PPE",
        variant="primary"
    )

    result = gr.Textbox(
        label="ผลการตรวจสอบ",
        lines=12
    )

    button.click(
        fn=check_object,
        inputs=[image, expected, api_key_input],
        outputs=result
    )

if __name__ == "__main__":
    # Render ต้องการให้แอปฟังที่ 0.0.0.0 และพอร์ตจากตัวแปร PORT
    # ไม่ใช้ share=True เพราะบน Render ไม่จำเป็นและทำให้เริ่มระบบช้า/ค้าง
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
