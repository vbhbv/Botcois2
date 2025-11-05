import torch
import os
import telebot
import soundfile as sf
import requests # مكتبة مطلوبة للتنزيل
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset 

# -------------------------------------------------------------
# 1. إعدادات البوت والنموذج
# -------------------------------------------------------------

# التوكن المضمَّن مباشرة (بناءً على طلبك ومسؤوليتك)
BOT_TOKEN = '6807502954:AAH5tOwXCjRXtF65wQFEDSkYeFBYIgUjblg' 

if not BOT_TOKEN:
    print("❌ خطأ فادح: التوكن غير موجود.")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# اسم مجلد النموذج المحلي (يجب أن يحتوي على الملفات الصغيرة)
MODEL_NAME = "./tts_model" 

# معرف الملف من رابط Google Drive
FILE_ID = "13Nq3fJslPv5gFgYxVV8bWE2mhbPor_yG"

# رابط التنزيل المباشر لملفات Google Drive الكبيرة
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# -------------------------------------------------------------
# 2. وظيفة التنزيل التلقائي لملف pytorch_model.bin
# -------------------------------------------------------------

# مكان حفظ ملف الأوزان
WEIGHTS_PATH = os.path.join(MODEL_NAME, "pytorch_model.bin")


def download_large_file_from_drive(url, target_path):
    """
    يقوم بتنزيل الملف الكبير من Google Drive إذا لم يكن موجوداً.
    """
    if os.path.exists(target_path):
        print(f"✅ الملف موجود بالفعل في: {target_path}")
        return

    print(f"⏳ تنزيل الملف الكبير (578MB) من Google Drive. قد يستغرق هذا وقتاً...")
    
    # التأكد من وجود مجلد tts_model قبل بدء التنزيل
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # تنزيل الملف مع متابعة إعادة التوجيه
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        token = get_confirm_token(response) # وظيفة مساعدة للتعامل مع رسائل التحذير من جوجل درايف

        if token:
            params = {'id': FILE_ID, 'export': 'download', 'confirm': token}
            response = session.get(url, params=params, stream=True)

        response.raise_for_status() # التأكد من نجاح الطلب
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768): # 32kb chunks
                f.write(chunk)
        print("✅ اكتمل التنزيل بنجاح.")
    except Exception as e:
        print(f"❌ فشل التنزيل من الرابط: {e}")

# وظيفة مساعدة للحصول على توكن تأكيد التحميل من جوجل درايف للملفات الكبيرة
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# -------------------------------------------------------------
# 3. تحميل النموذج والخطوط الصوتية
# -------------------------------------------------------------

# يتم تشغيل التنزيل قبل محاولة تحميل النموذج
download_large_file_from_drive(DOWNLOAD_URL, WEIGHTS_PATH)

print("⏳ جارٍ تهيئة النموذج والخطوط الصوتية...")

# تحميل الـ embeddings لخط متحدث افتراضي.
try:
    embeddings_dataset = load_dataset("microsoft/speecht5_tts", split="train")
    speaker_embeddings = torch.tensor(embeddings_dataset[5105]["xvector"]).unsqueeze(0)
    print("✅ تم تحميل الخطوط الصوتية بنجاح.")
except Exception as e:
    print(f"❌ فشل تحميل الخطوط الصوتية: {e}")
    speaker_embeddings = None

# إعداد الـ Pipeline من المجلد المحلي
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
    print(f"✅ تم تحميل نموذج TTS بنجاح: '{MODEL_NAME}'.")
except Exception as e:
    print(f"❌ فشل تحميل نموذج TTS من المسار المحلي. يرجى التأكد من اكتمال الملفات: {e}")
    synthesiser = None


# -------------------------------------------------------------
# 4. دالة توليد الصوت (لم تتغير)
# -------------------------------------------------------------

def text_to_audio(text_input, output_filename="output.ogg"):
    """
    تحول النص العربي إلى ملف صوتي باستخدام نموذج SpeechT5.
    """
    if not synthesiser or speaker_embeddings is None:
        return None 

    print(f"-> توليد الصوت للنص: '{text_input[:30]}...'")
    
    # تشغيل عملية التوليد
    speech = synthesiser(
        text_input,
        forward_params={"speaker_embeddings": speaker_embeddings}
    )

    # حفظ ملف الصوت بصيغة OGG (موصى بها لتليجرام)
    sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
    
    return output_filename

# -------------------------------------------------------------
# 5. وظائف بوت تليجرام وتشغيله (لم تتغير)
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

    # إرسال رسالة حالة (قيد المعالجة)
    status_message = bot.reply_to(message, "⏳ جارٍ معالجة النص...")

    try:
        # توليد ملف الصوت
        output_file_name = f"audio_{message.chat.id}.ogg"
        audio_file_path = text_to_audio(user_text, output_file_name)
        
        if audio_file_path:
            # إرسال الملف الصوتي ثم حذفه
            with open(audio_file_path, 'rb') as audio_file:
                bot.send_voice(message.chat.id, audio_file)
            
            os.remove(audio_file_path)
            
        else:
            bot.edit_message_text("❌ عذراً، لم يتمكن البوت من توليد الصوت. تأكد من تحميل الملفات.", status_message.chat.id, status_message.message_id)

    except Exception as e:
        print(f"❌ حدث خطأ أثناء المعالجة: {e}")
        bot.edit_message_text("❌ حدث خطأ غير متوقع أثناء معالجة طلبك.", status_message.chat.id, status_message.message_id)

    # حذف رسالة الحالة
    try:
        bot.delete_message(status_message.chat.id, status_message.message_id)
    except Exception:
        pass 

print("🚀 بدء تشغيل البوت...")
try:
    bot.infinity_polling()
except Exception as e:
    print(f"❌ فشل تشغيل البوت: {e}")
