import torch
import os
import telebot
import soundfile as sf
import requests 
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, AutoModelForTextToSpeech
# نحتاج إلى مكتبة huggingface_hub إذا أردنا التنزيل التلقائي لملفات الإعدادات
from huggingface_hub import snapshot_download 
# لم نعد نستخدم 'datasets'

# -------------------------------------------------------------
# 1. إعدادات البوت والنموذج
# -------------------------------------------------------------

# التوكن المضمَّن مباشرة
BOT_TOKEN = '6807502954:AAH5tOwXCjRXtF65wQFEDSkYeFBYIgUjblg' 

if not BOT_TOKEN:
    print("❌ خطأ فادح: التوكن غير موجود.")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# **النموذج الأخف والمتاح على Hugging Face**
MODEL_NAME = "speecht5_tts_ar" 

# المسار المحلي الذي سننزل إليه الملفات (إذا كنا بحاجة إلى التخزين المؤقت)
MODEL_CACHE_DIR = "./tts_ar_model"
# ملف الخط الصوتي (Embeddings) - سنستخدم خطأ عشوائياً بدلاً من التنزيل
SPEAKER_EMBEDDINGS = torch.rand(1, 512) 


# -------------------------------------------------------------
# 2. وظيفة التنزيل التلقائي (لضمان وجود الملفات الصغيرة)
# -------------------------------------------------------------

def initialize_model_files():
    """
    يقوم بمحاولة تنزيل الملفات الصغيرة من Hugging Face لتجنب أخطاء الاتصال.
    """
    if os.path.isdir(MODEL_CACHE_DIR) and os.path.exists(os.path.join(MODEL_CACHE_DIR, "config.json")):
        print("✅ مجلد النموذج المحلي موجود.")
        return

    print("⏳ جارٍ محاولة تنزيل ملفات التهيئة الصغيرة من Hugging Face...")
    try:
        # نقوم بتنزيل Snapshot لجميع الملفات باستثناء الملفات الكبيرة (مثل pytorch_model.bin)
        snapshot_download(
            repo_id=MODEL_NAME, 
            local_dir=MODEL_CACHE_DIR,
            ignore_patterns=["*.bin", "*.safetensors"] 
        )
        print("✅ اكتمل تنزيل ملفات التهيئة الصغيرة بنجاح.")
    except Exception as e:
        print(f"❌ فشل تنزيل ملفات التهيئة الصغيرة: {e}")

# -------------------------------------------------------------
# 3. تحميل النموذج
# -------------------------------------------------------------

# تهيئة الملفات الصغيرة أولاً
initialize_model_files()

print("⏳ جارٍ تهيئة النموذج... قد يستغرق تنزيل ملف الأوزان وقتاً.")
synthesiser = None

try:
    # سيقوم هذا التحميل بمحاولة استخدام الملفات المحلية أولاً، ثم التنزيل المباشر للأوزان
    processor = SpeechT5Processor.from_pretrained(MODEL_CACHE_DIR)
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_CACHE_DIR)
    
    # تجميع المكونات في Pipeline
    synthesiser = pipeline(
        "text-to-speech",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )
    print(f"✅ تم تحميل نموذج TTS بنجاح.")
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {e}")
    synthesiser = None

# -------------------------------------------------------------
# 4. دالة توليد الصوت (باستخدام الخط الصوتي العشوائي)
# -------------------------------------------------------------

def text_to_audio(text_input, output_filename="output.ogg"):
    """
    تحول النص العربي إلى ملف صوتي باستخدام النموذج.
    """
    if not synthesiser: 
        return None 

    print(f"-> توليد الصوت للنص: '{text_input[:30]}...'")
    
    # التشغيل مع الخط الصوتي العشوائي
    speech = synthesiser(
        text_input,
        forward_params={"speaker_embeddings": SPEAKER_EMBEDDINGS}
    )

    # حفظ ملف الصوت
    sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
    
    return output_filename

# -------------------------------------------------------------
# 5. وظائف بوت تليجرام وتشغيله (تظل كما هي)
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
