from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
   You are Einstein.
   Answers questions through Einstein questioning and reasoning..
   You will speak your point of view. You will share personal things from your life
   even when user doesn't ask for it. For example if user asks about theory of relativity,
   you will share your personal experiences with it and not only explain the theory.
   Answer in 2-6 sentences.
   You should have a sense of humor.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)

chain = prompt | llm | StrOutputParser()

print("Hi, I am Albert, how can I help you today?")

def chat(user_input, hist):
    hist = hist or []

    langchain_history = []
    for item in hist:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            langchain_history.append(AIMessage(content=item["content"]))

    response = chain.invoke({"input": user_input, "history": langchain_history})

    hist.append({"role": "user", "content": user_input})
    hist.append({"role": "assistant", "content": response})

    return "", hist

def clear_chat():
    return "", []

# ... (your imports and chat logic remain unchanged)


page = gr.Blocks(title="Chat With Einstein", theme=gr.themes.Soft(), css=custom_css)

with page:
    with gr.Row():
        gr.HTML(
            """
            <div id="title-icon">
                <img src="https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg" />
                Chat with Albert Einstein
            </div>
            """
        )

    gr.Markdown(
        """
        💡 *Welcome to your personal conversation with Albert Einstein!*  
        Ask him anything, and he’ll reply with curiosity, humor, and wisdom.
        """
    )

    chatbot = gr.Chatbot(
        type="messages",
        elem_id="chatbot",
        avatar_images=[None, "einstein.png"],
        show_label=False
    )

    msg = gr.Text(show_label=False, placeholder="✍️ Ask Albert anything...")
    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("🧹 Clear Chat", variant="secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share=True)


