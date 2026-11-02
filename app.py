import os
import time
import logging
import gradio as gr
from google import genai
from google.genai import types
from PIL import Image

logging.basicConfig(level=logging.INFO)
# ใช้ชื่อโมเดลปัจจุบันเป็นค่าเริ่มต้น; Render Environment GEMINI_MODEL สามารถ override ได้
MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash').strip() or 'gemini-2.5-flash'
PPE_ITEMS = ['ถุงมือ', 'แว่นตานิรภัย', 'เสื้อช็อป', 'สายสะท้อนแสง', 'หน้ากากอนามัย']

PROMPT = '''คุณเป็น AI ผู้ช่วยตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE)
สำหรับห้องปฏิบัติการเคมี

กรุณาวิเคราะห์ภาพที่แนบมาอย่างละเอียด
โดยตรวจสอบเฉพาะอุปกรณ์ต่อไปนี้: {items}

ข้อกำหนดในการตรวจสอบ:
1. ถุงมือ
- ตรวจสอบถุงมือทุกสี เช่น สีขาว น้ำเงิน ม่วง เหลือง ดำ เขียว ชมพู และสีอื่น ๆ
- ตรวจสอบทั้งถุงมือยางและถุงมือไนไตรล์ และว่ามีการสวมที่มือจริงหรือไม่
- ห้ามตัดสินว่าถุงมือไม่พบเพียงเพราะสีแตกต่างจากสีขาว
2. แว่นตานิรภัย
- ตรวจสอบแว่นครอบตาและแว่นตานิรภัย
- พิจารณาจากรูปทรง เลนส์ และลักษณะของแว่น
- อย่าสรุปว่าแว่นสายตาทั่วไปเป็นแว่นนิรภัยโดยไม่มีหลักฐาน
3. เสื้อช็อป
- ตรวจสอบเสื้อช็อปวิศวกรรม เสื้อเครื่องแบบช่าง และเสื้อปฏิบัติงาน
- พิจารณาจากรูปแบบเสื้อและลักษณะเครื่องแต่งกาย
- หากไม่สามารถยืนยันได้ว่าเป็นชุดช็อปจริง ให้ระบุว่าไม่สามารถยืนยันได้
4. สายสะท้อนแสง
- ตรวจสอบแถบสะท้อนแสงบนเสื้อหรือเครื่องแต่งกาย
- ไม่จำกัดสี เช่น สีเงิน สีเหลือง หรือสีส้ม
- สีสดเพียงอย่างเดียวไม่ยืนยันคุณสมบัติสะท้อนแสง
5. หน้ากากอนามัย
- ตรวจสอบหน้ากากที่สวมปิดจมูกและปาก
- ตรวจสอบทุกสี เช่น สีขาว ฟ้า เขียว ดำ และสีอื่น ๆ

หลักการสำคัญ:
- ตรวจสอบจากรูปร่าง ลักษณะ และหน้าที่ของอุปกรณ์ ไม่ใช้สีเป็นเงื่อนไขตัดสินเพียงอย่างเดียว
- ตรวจสอบทุกรายการที่ผู้ใช้เลือกเท่านั้น หากพบหลายประเภทให้รายงานทุกประเภท
- ห้ามแต่งข้อมูลหรือคาดเดาว่ามีอุปกรณ์ หากมองไม่เห็นชัดเจน
- หากภาพไม่ชัดให้ระบุว่าไม่สามารถยืนยันได้

กรุณาตอบเป็นภาษาไทยในรูปแบบดังนี้:
ผลการตรวจสอบ PPE
- ชื่ออุปกรณ์: พบ / ไม่พบ / ไม่สามารถยืนยันได้
  รายละเอียด: อธิบายสี ลักษณะ และสิ่งที่สังเกตเห็น

สรุปผลการตรวจสอบ:
อธิบายว่าอุปกรณ์ใดพบ ไม่พบ และไม่สามารถยืนยันได้

หมายเหตุ: การวิเคราะห์จากภาพเป็นเพียงการประเมินเบื้องต้น ไม่ใช่การรับรองความปลอดภัยในการเข้าห้องปฏิบัติการ และไม่สามารถรับรองมาตรฐานอุปกรณ์จากภาพเพียงอย่างเดียว
'''

def check_object(image, expected_list, api_key):
    api_key = (api_key or '').strip()
    if not api_key:
        return 'กรุณากรอก Gemini API Key ก่อนค่ะ'
    if image is None:
        return 'กรุณาอัปโหลดหรือถ่ายภาพก่อนค่ะ'
    if not expected_list:
        return 'กรุณาเลือกอุปกรณ์ PPE อย่างน้อย 1 รายการค่ะ'
    selected = [item for item in PPE_ITEMS if item in expected_list]
    if not selected:
        return 'ไม่มีรายการ PPE ที่รองรับ'
    try:
        img = image.convert('RGB')
        img.thumbnail((1600, 1600))
        with genai.Client(api_key=api_key) as client:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[PROMPT.format(items=', '.join(selected)), img],
                        config=types.GenerateContentConfig(temperature=0.1),
                    )
                    return (response.text or '').strip() or 'AI ไม่ส่งผลการวิเคราะห์กลับมา กรุณาลองใหม่อีกครั้ง'
                except Exception as exc:
                    code = getattr(exc, 'code', None)
                    message = str(exc)
                    upper = message.upper()
                    if code is None:
                        for candidate in (400, 401, 403, 404, 429, 500, 502, 503, 504):
                            if str(candidate) in message:
                                code = candidate
                                break
                    if code in (429, 500, 502, 503, 504) or 'UNAVAILABLE' in upper or 'RESOURCE_EXHAUSTED' in upper:
                        if attempt < 2:
                            time.sleep(2 * (attempt + 1))
                            continue
                    if code == 404 or 'NOT_FOUND' in upper:
                        return f'ไม่พบโมเดล {MODEL} สำหรับ API Key นี้ กรุณาตรวจสอบตัวแปร GEMINI_MODEL ใน Render หรือรายชื่อโมเดลที่รองรับใน Google AI Studio'
                    if code == 429 or 'RESOURCE_EXHAUSTED' in upper:
                        return 'ใช้งาน API เกินโควตาหรือส่งคำขอถี่เกินไป กรุณารอสักครู่แล้วลองใหม่อีกครั้ง'
                    if code in (401, 403) or 'API_KEY_INVALID' in upper or 'PERMISSION_DENIED' in upper:
                        return 'API Key ไม่ถูกต้องหรือไม่มีสิทธิ์ใช้งาน กรุณาตรวจสอบ API Key ใน Google AI Studio'
                    if code == 400 or 'INVALID_ARGUMENT' in upper:
                        return f'คำขอไม่ถูกต้อง (400): {message}'
                    if code in (500, 502, 503, 504) or 'UNAVAILABLE' in upper:
                        return 'เซิร์ฟเวอร์ Gemini ไม่พร้อมใช้งานชั่วคราว กรุณาลองใหม่อีกครั้งในภายหลัง'
                    logging.exception('Gemini API error')
                    return f'เกิดข้อผิดพลาดจาก Gemini API: {message}'
        return 'ไม่สามารถเชื่อมต่อ Gemini ได้หลังจากลองใหม่หลายครั้ง กรุณาลองใหม่อีกครั้ง'
    except Exception as exc:
        logging.exception('Unexpected error')
        return f'เกิดข้อผิดพลาด: {exc}'


with gr.Blocks(title='ตรวจสอบ PPE ห้องปฏิบัติการเคมี') as demo:
    gr.Markdown('# 🧪 ระบบตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE)\nอัปโหลดหรือถ่ายภาพ แล้วเลือกอุปกรณ์ที่ต้องการตรวจสอบ')
    with gr.Row():
        with gr.Column():
            api_key_input = gr.Textbox(label='Gemini API Key', type='password', placeholder='วาง API Key ที่นี่')
            image_input = gr.Image(type='pil', label='ภาพผู้ใช้งาน', sources=['upload', 'webcam'])
            items_input = gr.CheckboxGroup(choices=PPE_ITEMS, value=PPE_ITEMS, label='อุปกรณ์ที่ต้องการตรวจสอบ')
            submit_btn = gr.Button('ตรวจสอบ', variant='primary')
        with gr.Column():
            output = gr.Textbox(label='ผลการตรวจสอบ', lines=20)
    submit_btn.click(check_object, inputs=[image_input, items_input, api_key_input], outputs=output)


if __name__ == '__main__':
    port = int(os.getenv('PORT', '7860'))
    demo.queue().launch(server_name='0.0.0.0', server_port=port)
