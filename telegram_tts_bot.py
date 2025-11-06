import torch
import os
import telebot
import soundfile as sf
import requests 
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech
# لم نعد نستخدم 'datasets' لذلك تم حذفها

# -------------------------------------------------------------
# 1. إعدادات البوت والنموذج
# -------------------------------------------------------------

# التوكن المضمَّن مباشرة
BOT_TOKEN = '6807502954:AAH5tOwXCjRXtF65wQFEDSkYeFBYIgUjblg' 

if not BOT_TOKEN:
    print("❌ خطأ فادح: التوكن غير موجود.")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# اسم مجلد النموذج المحلي (يجب أن يحتوي على جميع الملفات الصغيرة)
MODEL_NAME = "./tts_model" 

# معرف ملف pytorch_model.bin من Google Drive 
FILE_ID = "13Nq3fJslPv5gFgYxVV8bWE2mhbPor_yG"

# رابط التنزيل المباشر
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# -------------------------------------------------------------
# 2. وظيفة التنزيل التلقائي لملف pytorch_model.bin
# -------------------------------------------------------------

WEIGHTS_PATH = os.path.join(MODEL_NAME, "pytorch_model.bin")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_large_file_from_drive(url, target_path):
    """
    يقوم بتنزيل الملف الكبير من Google Drive إذا لم يكن موجوداً.
    """
    if os.path.exists(target_path):
        print(f"✅ ملف pytorch_model.bin موجود بالفعل.")
        return

    print(f"⏳ تنزيل الملف الكبير (578MB) من Google Drive. قد يستغرق هذا وقتاً...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': FILE_ID, 'export': 'download', 'confirm': token}
            response = session.get(url, params=params, stream=True)

        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768): 
                f.write(chunk)
        print("✅ اكتمل التنزيل بنجاح.")
    except Exception as e:
        print(f"❌ فشل التنزيل من الرابط: {e}")

# -------------------------------------------------------------
# 3. تحميل النموذج والخطوط الصوتية (تم حذف الخط الصوتي لتجاوز المشاكل)
# -------------------------------------------------------------

# تشغيل التنزيل قبل محاولة تحميل النموذج
download_large_file_from_drive(DOWNLOAD_URL, WEIGHTS_PATH)

print("⏳ جارٍ تهيئة النموذج...")

# بما أننا لا نستطيع تحميل ملف الخط الصوتي، سنقوم بتعيينه None
# وسيعتمد النموذج على خط صوتي داخلي أو افتراضي
speaker_embeddings = None

try:
    # 1. تحميل المعالج (Processor) من المجلد المحلي
    processor = SpeechT5Processor.from_pretrained(MODEL_NAME)
    # 2. تحميل الموديل (Model Weights) من المجلد المحلي
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME)
    
    # 3. تجميع المكونات في Pipeline للاستخدام السهل
    synthesiser = pipeline(
        "text-to-speech",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )
    print(f"✅ تم تحميل نموذج TTS بنجاح من المسار المحلي: '{MODEL_NAME}'.")
except Exception as e:
    print(f"❌ فشل تحميل النموذج من المسار المحلي. تأكد من وجود الملفات الصغيرة مثل preprocessor_config.json: {e}")
    synthesiser = None

# -------------------------------------------------------------
# 4. دالة توليد الصوت
# -------------------------------------------------------------

def text_to_audio(text_input, output_filename="output.ogg"):
    """
    تحول النص العربي إلى ملف صوتي باستخدام نموذج SpeechT5.
    """
    if not synthesiser: 
        return None 

    print(f"-> توليد الصوت للنص: '{text_input[:30]}...'")
    
    # تشغيل عملية التوليد. (تمرير الخط الصوتي فقط إذا كان موجوداً)
    if speaker_embeddings is not None:
        speech = synthesiser(
            text_input,
            forward_params={"speaker_embeddings": speaker_embeddings}
        )
    else:
        # التشغيل بدون خط صوتي محدد
        speech = synthesiser(text_input)

    # حفظ ملف الصوت
    sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
    
    return output_filename

# -------------------------------------------------------------
# 5. وظائف بوت تليجرام وتشغيله
# -------------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "👋 مرحباً! أرسل لي أي نص عربي وسأقوم بتحويله إلى مقطع صوتي باستخدام نموذج AI.")

@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_text = message.text
    
    if len(user_text) > 500: 
        bot.reply_to(message, "⚠️ عذراً، يرجى إرسال نص أقل من 500 حرف لتجنب المعالجة الطويلة.")
        return

    status_message = bot.reply_to(message, "⏳ جارٍ معالجة النص...")

    try:
        output_file_name = f"audio_{message.chat.id}.ogg"
        audio_file_path = text_to_audio(user_text, output_file_name)
        
        if audio_file_path:
            with open(audio_file_path, 'rb') as audio_file:
                bot.send_voice(message.chat.id, audio_file)
            
            os.remove(audio_file_path)
            
        else:
            bot.edit_message_text("❌ عذراً، فشل البوت في توليد الصوت. تأكد من تحميل جميع الملفات.", status_message.chat.id, status_message.message_id)

    except Exception as e:
        print(f"❌ حدث خطأ أثناء المعالجة: {e}")
        bot.edit_message_text("❌ حدث خطأ غير متوقع أثناء معالجة طلبك.", status_message.chat.id, status_message.message_id)

    try:
        bot.delete_message(status_message.chat.id, status_message.message_id)
    except Exception:
        pass 

print("🚀 بدء تشغيل البوت...")
try:
    bot.infinity_polling()
except Exception as e:
    print(f"❌ فشل تشغيل البوت: {e}")
