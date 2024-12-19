import gradio as gr
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pyngrok import ngrok
from deep_translator import GoogleTranslator  # Mengimpor GoogleTranslator

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mengambil kunci API GROQ dari variabel lingkungan
groq_api_key = os.getenv('GROQ_API_KEY')

# Validasi apakah kunci API tersedia
if not groq_api_key:
    raise ValueError("GROQ_API_KEY tidak ditemukan. Pastikan sudah diatur di file .env.")

# Fungsi untuk inisialisasi percakapan menggunakan model LLM GROQ
def initialize_conversation():
    memory = ConversationBufferWindowMemory(k=5)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=1,
    )
    return ConversationChain(llm=groq_chat, memory=memory)

conversation = initialize_conversation()

def chatbot(user_question, temperature=1):
    try:
        conversation.llm.temperature = temperature
        response = conversation(user_question)
        return [(user_question, response['response'])]
    except Exception as e:
        return [(user_question, f"Terjadi error: {e}")]

def reset_conversation():
    global conversation
    conversation = initialize_conversation()
    return []

# Fungsi untuk terjemahan
def translate_text(text, source_lang, target_lang):
    lang_code = {
        "Indonesian": "id",
        "English": "en",
        "Japanese": "ja",
        "Chinese": "zh"
    }
    try:
        translated = GoogleTranslator(source=lang_code[source_lang], target=lang_code[target_lang]).translate(text)
        return translated
    except Exception as e:
        return f"Error: {e}"

def create_interface():
    with gr.Blocks() as demo:
        # Judul dan Deskripsi
        gr.HTML("<h1 style='text-align:center; color: #4a90e2;'>Brivixel AI Chatbot</h1>")
        gr.HTML("<p style='text-align:center; color: #888;'>Tanya apa saja, chatbot akan memberikan jawaban terbaik!</p>")

        # Tema mode gelap/terang
        with gr.Row():
            with gr.Column():
                theme_toggle = gr.Checkbox(value=False, label="Mode Gelap", interactive=True)
        
        # Area Percakapan
        with gr.Row():
            with gr.Column():
                chat_history = gr.Chatbot()  # Komponen chat
                message_input = gr.Textbox(placeholder="Ketik pertanyaan Anda di sini...", label="Pertanyaan", lines=2)
                temperature_slider = gr.Slider(minimum=0, maximum=2, value=1, step=0.1, label="Suhu Respons (Creativity)")

        # Tombol kirim dan reset
        with gr.Row():
            submit_button = gr.Button("Kirim 📨", elem_id="send-btn", variant="primary")
            reset_button = gr.Button("Reset 🔄", elem_id="reset-btn", variant="secondary")

        submit_button.click(
            chatbot, 
            inputs=[message_input, temperature_slider],
            outputs=[chat_history]
        )

        reset_button.click(reset_conversation, outputs=[chat_history])

        # Fitur Terjemahan
        with gr.Row():
            source_language = gr.Dropdown(choices=["Indonesian", "English", "Japanese", "Chinese"], label="Bahasa Sumber", value="Indonesian")
            target_language = gr.Dropdown(choices=["Indonesian", "English", "Japanese", "Chinese"], label="Bahasa Target", value="English")
            text_to_translate = gr.Textbox(placeholder="Masukkan teks untuk diterjemahkan...", label="Teks untuk Terjemahan", lines=2)
            translation_output = gr.Textbox(label="Hasil Terjemahan", lines=2)

        # Tombol Terjemahkan
        translate_button = gr.Button("Terjemahkan 🌐", elem_id="translate-btn", variant="secondary")

        translate_button.click(
            translate_text, 
            inputs=[text_to_translate, source_language, target_language], 
            outputs=[translation_output]
        )

        # Mengubah tema saat mode gelap dipilih
        theme_toggle.change(
            lambda x: "dark" if x else "light",
            inputs=[theme_toggle],
            outputs=[demo]
        )

    return demo

# Menjalankan aplikasi Gradio dengan ngrok
if __name__ == "__main__":
    interface = create_interface()  # Membuat antarmuka
    
    # Membuka tunnel ngrok untuk port 7860
    public_url = ngrok.connect(7860)
    print(f"Gradio app is live at: {public_url}")
    
    interface.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
