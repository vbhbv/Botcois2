import torch
from transformers import pipeline
import soundfile as sf
import os
import telebot
from datasets import load_dataset # نحتاجها لتحميل Speaker Embeddings الافتراضية

# -------------------------------------------------------------
# 1. إعدادات البوت والنموذج
# -------------------------------------------------------------

# *** يجب عليك استبدال هذا التوكن بتوكن البوت الخاص بك من BotFather ***
BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN' 
bot = telebot.TeleBot(BOT_TOKEN)

# اسم المستودع الصحيح الذي سيتم تنزيله تلقائيًا
MODEL_NAME = "MBZUAI/speecht5_tts_claritts_ar"

# -------------------------------------------------------------
# 2. تحميل الخطوط الصوتية (Speaker Embeddings) والنموذج
# -------------------------------------------------------------

print("Initializing Model and Speaker Embeddings...")

# تحميل الـ embeddings لخط متحدث افتراضي (هذا الإجراء يتم مرة واحدة)
# يمكنك تغيير الرقم (5105) للحصول على نبرة صوت مختلفة قليلاً
try:
    embeddings_dataset = load_dataset("microsoft/speecht5_tts", split="train")
    # نستخدم الخط الصوتي لرقم 5105 كمثال
    speaker_embeddings = torch.tensor(embeddings_dataset[5105]["xvector"]).unsqueeze(0)
    print("Speaker Embeddings loaded successfully.")
except Exception as e:
    print(f"Error loading Speaker Embeddings dataset: {e}")
    speaker_embeddings = None # إذا فشل التحميل، يجب التعامل مع هذا لاحقاً

# إعداد الـ Pipeline لنموذج تحويل النص إلى كلام العربي
try:
    synthesiser = pipeline(
        "text-to-speech", 
        MODEL_NAME
    )
    print(f"TTS Model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading TTS Model: {e}")
    synthesiser = None

# -------------------------------------------------------------
# 3. دالة توليد الصوت
# -------------------------------------------------------------

def text_to_audio(text_input, output_filename="output.ogg"):
    """
    تحول النص العربي إلى ملف صوتي باستخدام نموذج SpeechT5.
    """
    if not synthesiser or speaker_embeddings is None:
        return None # لا يمكن توليد الصوت بدون النموذج أو Embeddings

    print(f"Generating audio for: '{text_input}'")
    
    # تشغيل عملية التوليد
    speech = synthesiser(
        text_input,
        forward_params={"speaker_embeddings": speaker_embeddings}
    )

    # حفظ ملف الصوت بصيغة OGG (صيغة مفضلة لبوتات تليجرام)
    sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
    
    return output_filename

# -------------------------------------------------------------
# 4. وظائف بوت تليجرام
# -------------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "مرحباً بك! أنا بوت لتحويل النص العربي إلى كلام. أرسل لي أي نص وسأقوم بتحويله إلى مقطع صوتي.")

@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_text = message.text
    
    if len(user_text) > 500: # حد أقصى للنص لتجنب المعالجة الطويلة جداً
        bot.reply_to(message, "عذراً، يرجى إرسال نص أقل من 500 حرف.")
        return

    # إرسال رسالة "جارٍ المعالجة..." لتجنب انتظار المستخدم
    status_message = bot.reply_to(message, "⏳ جارٍ معالجة النص وتحويله إلى كلام، يرجى الانتظار...")

    try:
        # توليد ملف الصوت
        output_file_name = f"audio_{message.chat.id}.ogg"
        audio_file_path = text_to_audio(user_text, output_file_name)
        
        if audio_file_path:
            # إرسال الملف الصوتي
            with open(audio_file_path, 'rb') as audio_file:
                # نستخدم send_voice لأنها تتناسب مع ملفات OGG/Opus
                bot.send_voice(message.chat.id, audio_file, caption=f"تم توليد الصوت بنجاح.")
            
            # تنظيف الملف
            os.remove(audio_file_path)
            
        else:
            bot.edit_message_text("❌ عذراً، لم يتمكن البوت من توليد الصوت (قد تكون ملفات النموذج لم تكتمل).", 
                                  status_message.chat.id, status_message.message_id)

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.edit_message_text("❌ عذراً، حدث خطأ أثناء معالجة طلبك.", 
                              status_message.chat.id, status_message.message_id)

    # حذف رسالة الحالة بعد الانتهاء
    try:
        bot.delete_message(status_message.chat.id, status_message.message_id)
    except Exception:
        pass # قد لا يمكن حذفه إذا كان قد تم تعديله بالفعل

# -------------------------------------------------------------
# 5. تشغيل البوت
# -------------------------------------------------------------

print("Starting bot polling...")
try:
    # لتشغيل البوت بشكل مستمر
    bot.infinity_polling()
except Exception as e:
    print(f"Bot failed to start: {e}")
