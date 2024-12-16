# Import library yang dibutuhkan
import gradio as gr  # Gradio untuk antarmuka web
import os  # Mengakses variabel lingkungan
from langchain.chains import ConversationChain  # Untuk membuat rantai percakapan
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  # Memori percakapan
from langchain_groq import ChatGroq  # Model LLM "GROQ" untuk percakapan
from dotenv import load_dotenv  # Untuk memuat variabel lingkungan dari file .env

# Memuat variabel lingkungan dari file .env
load_dotenv()  # Membaca file .env yang menyimpan variabel sensitif seperti kunci API

# Mengambil kunci API GROQ dari variabel lingkungan
groq_api_key = os.environ['GROQ_API_KEY']

# Fungsi untuk inisialisasi percakapan menggunakan model LLM GROQ
def initialize_conversation():
    # Inisialisasi memori percakapan agar chatbot bisa "mengingat" konteks dialog sebelumnya
    memory = ConversationBufferWindowMemory(k=5)  # Menyimpan 5 percakapan terakhir

    # Membuat objek ChatGroq dengan API key, model, dan pengaturan
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,  # Menggunakan kunci API untuk autentikasi
        model_name="llama3-8b-8192",  # Nama model yang digunakan
        temperature=1,  # Kreativitas respons (semakin tinggi, semakin kreatif/jelas jawabannya)
    )

    # Mengembalikan objek ConversationChain yang menggabungkan model dan memori
    return ConversationChain(llm=groq_chat, memory=memory)

# Membuat objek conversation untuk mengelola alur percakapan
conversation = initialize_conversation()

# Fungsi untuk menangani pertanyaan dari pengguna
def chatbot(user_question, temperature=1):
    try:
        # Mengubah pengaturan suhu kreatifitas sesuai pilihan pengguna
        conversation.llm.temperature = temperature
        # Mengirim pertanyaan ke model dan mendapatkan respons
        response = conversation(user_question)
        # Mengembalikan respons dalam format yang kompatibel dengan Gradio
        return [(user_question, response['response'])]  # Format yang sesuai untuk Gradio Chatbot
    except Exception as e:  # Jika ada error
        # Mengembalikan pesan error agar mudah dilihat
        return [(user_question, f"Terjadi error: {e}")]

# Fungsi untuk memulai percakapan baru
def reset_conversation():
    global conversation
    conversation = initialize_conversation()  # Menyegarkan objek conversation untuk memulai percakapan baru
    return []  # Mengembalikan chat kosong untuk memulai percakapan baru

# Menyiapkan layout dan antarmuka Gradio dengan tampilan yang lebih menarik
def create_interface():
    with gr.Blocks() as demo:
        # Menambahkan elemen-elemen antarmuka
        gr.HTML("<h1 style='text-align:center; color: #4a90e2;'>Brivixel AI</h1>")
        gr.HTML("<p style='text-align:center; color: #888;'>Tanya apa saja, chatbot akan memberikan jawaban terbaik!</p>")

        # Area percakapan dengan chat history
        with gr.Row():
            with gr.Column():
                chat_history = gr.Chatbot()  # Komponen chatbot yang menampilkan riwayat percakapan
                message_input = gr.Textbox(placeholder="Ketik pertanyaan Anda di sini...", label="Pertanyaan")  # Input untuk pertanyaan pengguna
                temperature_slider = gr.Slider(minimum=0, maximum=2, value=1, step=0.1, label="Suhu Respons (Creativity)")  # Pengaturan suhu respons

        # Tombol untuk mengirim pertanyaan
        submit_button = gr.Button("Kirim")
        submit_button.click(
            chatbot, 
            inputs=[message_input, temperature_slider],  # Input pertanyaan dan suhu
            outputs=[chat_history]  # Output hasil respons chatbot
        )

        # Tombol untuk mereset percakapan
        reset_button = gr.Button("Mulai Percakapan Baru")
        reset_button.click(reset_conversation, outputs=[chat_history])

    return demo

# Menjalankan aplikasi Gradio
if __name__ == "__main__":
    interface = create_interface()  # Membuat antarmuka
    interface.launch(share=True, inbrowser=True)  # Menjalankan antarmuka di browser dan membagikan URL
