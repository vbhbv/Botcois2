import torch
import os
import telebot
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset 

# -------------------------------------------------------------
# 1. إعدادات البوت والنموذج
# -------------------------------------------------------------

# الحصول على التوكن من متغيرات البيئة (TELEGRAM_BOT_TOKEN)
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN') 

if not BOT_TOKEN:
    print("❌ خطأ فادح: متغير البيئة TELEGRAM_BOT_TOKEN غير مضبوط.")
    # الخروج من البرنامج إذا لم يتم العثور على التوكن
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# اسم مستودع النموذج العربي لتحويل النص إلى كلام
MODEL_NAME = "MBZUAI/speecht5_tts_claritts_ar"

# -------------------------------------------------------------
# 2. تحميل الخطوط الصوتية (Speaker Embeddings) والنموذج
# -------------------------------------------------------------

print("⏳ جارٍ تهيئة النموذج والخطوط الصوتية...")

# تحميل الـ embeddings لخط متحدث افتراضي.
try:
    # هذا السطر سيقوم بتنزيل مجموعة البيانات المطلوبة للمرة الأولى
    embeddings_dataset = load_dataset("microsoft/speecht5_tts", split="train")
    # نستخدم الخط الصوتي لرقم 5105 كمثال لنبرة الصوت
    speaker_embeddings = torch.tensor(embeddings_dataset[5105]["xvector"]).unsqueeze(0)
    print("✅ تم تحميل الخطوط الصوتية بنجاح.")
except Exception as e:
    print(f"❌ فشل تحميل الخطوط الصوتية: {e}")
    speaker_embeddings = None

# إعداد الـ Pipeline (هذا السطر سيقوم بتنزيل ملفات النموذج للمرة الأولى)
try:
    synthesiser = pipeline(
        "text-to-speech", 
        MODEL_NAME
    )
    print(f"✅ تم تحميل نموذج TTS بنجاح: '{MODEL_NAME}'.")
except Exception as e:
    print(f"❌ فشل تحميل نموذج TTS: {e}")
    synthesiser = None

# -------------------------------------------------------------
# 3. دالة توليد الصوت
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
# 4. وظائف بوت تليجرام
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
            bot.edit_message_text("❌ عذراً، لم يتمكن البوت من توليد الصوت.", status_message.chat.id, status_message.message_id)

    except Exception as e:
        print(f"❌ حدث خطأ أثناء المعالجة: {e}")
        bot.edit_message_text("❌ حدث خطأ غير متوقع أثناء معالجة طلبك.", status_message.chat.id, status_message.message_id)

    # حذف رسالة الحالة
    try:
        bot.delete_message(status_message.chat.id, status_message.message_id)
    except Exception:
        pass 

# -------------------------------------------------------------
# 5. تشغيل البوت
# -------------------------------------------------------------

print("🚀 بدء تشغيل البوت...")
try:
    bot.infinity_polling()
except Exception as e:
    print(f"❌ فشل تشغيل البوت: {e}")
