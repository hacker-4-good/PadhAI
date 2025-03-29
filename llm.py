#--------------------------------------------------------------DSPY RAG---------------------------------------------------------------

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction 
import dspy 
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from pydantic import BaseModel, Field
load_dotenv()


llm = dspy.PremAI(project_id=8883, api_key='INSERT API-KEY', temperature=1)
rm = ChromadbRM(
    collection_name = "Algorithm", 
    persist_directory = "DB", 
    embedding_function = SentenceTransformerEmbeddingFunction()
)

dspy.configure(lm = llm, rm=rm)

class QuerySignature(dspy.Signature):
    '''
    Provide complete and to-the-point answers to student queries regarding their subjects, including both theoretical questions and numerical problems, using content from textbooks.
    *You are great in mathematics so show proper steps to solve numericals*
    '''
    context = dspy.InputField(desc="may contain relevant facts from textbooks")
    question: str = dspy.InputField(desc="Student's question, either theoretical or numerical")
    answer: str = dspy.OutputField(desc="Complete and to-the-point answer")

class RAG(dspy.Module):
    def __init__(self, num_passage=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passage)
        self.generate_answer = dspy.ChainOfThought(signature=QuerySignature)

    def forward(self, question):
        context = self.retrieve(question).passages
        print(context)
        prediction = self.generate_answer(context = context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

#-------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------WEB APPLICATION---------------------------------------------------------------
import streamlit as st 
import os
from icrawler.builtin import GoogleImageCrawler
import nltk 
import time

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

google_Crawler = GoogleImageCrawler(storage = {'root_dir': 'Images'})

st.set_page_config(layout="wide", page_title="PadhAI", page_icon=":books:")

st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

    body {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background: #fff;
        margin: 0;
        flex-direction: column;
    }

    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        flex-direction: column;
    }

    .book {
        width: 50px;
        height: 60px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .book::before, .book::after {
        content: '';
        width: 25px;
        height: 60px;
        background: #00d4c0;
        position: absolute;
        border-radius: 5px;
        border: 3px solid #0066ff;
    }

    .book::before {
        left: 0;
        transform: skewY(-10deg);
    }

    .book::after {
        right: 0;
        transform: skewY(10deg);
    }

    .text {
        font-family: 'Pacifico', cursive;
        font-size: 48px;
        color: var(--text-color);
        margin-top: 20px;
    }

    .subtitle {
        font-size: 14px;
        color: var(--text-color);
        text-align: center;
        margin-top: 5px;
    }
</style>

<script>
    const setTheme = (theme) => {
        document.documentElement.style.setProperty('--text-color', theme === 'dark' ? '#fff' : '#333');
    }

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                const theme = document.body.classList.contains('dark') ? 'dark' : 'light';
                setTheme(theme);
            }
        });
    });

    observer.observe(document.body, { attributes: true });

    // Initial theme set
    const initialTheme = document.body.classList.contains('dark') ? 'dark' : 'light';
    setTheme(initialTheme);
</script>

<div class="logo-container">
    <div class="book"></div>
    <div class="text">PadhAI</div>
</div>
<div class="subtitle">REVOLUTIONIZING EDUCATION, ONE LESSON AT A TIME</div>
''', unsafe_allow_html=True)

s1, s2 = st.columns([3,1])

if 'history' not in st.session_state:
    st.session_state.history = []

def get_response(question):
    response = RAG().forward(question=question)
    return response

with s1:
    st.header("Ask Me 💭")
    question = st.chat_input("Ask a question:")
    if question:
        with st.spinner('Waiting for response...'):
            response = get_response(question)
            time.sleep(2)  # Simulate waiting time for LLM response
        st.session_state.history.append((question, response.answer))

    if st.session_state.history:
        for q, r in st.session_state.history:
            with st.chat_message("user"):
                st.write(f"**Question:** {q}")
            with st.chat_message("assistant"):
                st.write(f"**Answer:** {r}")

    st.header("Quiz 📜")
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = None
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = {"correct": 0, "total": 0}

    user_topic = st.text_input("Enter the topic for the quiz:")
    if user_topic and st.session_state.quiz_state is None:
        # Retrieve relevant context from ChromaDB
        context = dspy.Retrieve()(user_topic).passages

        class QuizInput(BaseModel):
            topic: str = Field(description="The topic for the quiz")
            context: list[str] = Field(description="Relevant context from ChromaDB")

        class QuizOption(BaseModel):
            option: str = Field(description="A possible answer option")

        class QuizOutput(BaseModel):
            question: str = Field(description="The generated quiz question")
            options: list[QuizOption] = Field(description="The list of answer options")
            correct_option: int = Field(ge=0, le=3, description="The index of the correct answer option")

        class QuizSignature(dspy.Signature):
            """Generate a quiz question on a user-provided topic with 4 answer options."""
            input: QuizInput = dspy.InputField()
            output: QuizOutput = dspy.OutputField()

        predictor = dspy.TypedPredictor(QuizSignature)
        quiz_input = QuizInput(topic=user_topic, context=context)
        prediction = predictor(input=quiz_input)

        st.session_state.quiz_state = {
            "question": prediction.output.question,
            "options": [option.option for option in prediction.output.options],
            "correct_option_index": prediction.output.correct_option,
            "topic": user_topic
        }

    if st.session_state.quiz_state:
        quiz_state = st.session_state.quiz_state
        st.write(f"**Quiz Topic:** {quiz_state['topic']}")
        st.write(f"**Question:** {quiz_state['question']}")
        selected_option = st.radio("Select an option:", quiz_state['options'], key="quiz_option")
        if st.button("Check Answer"):
            st.session_state.quiz_score["total"] += 1
            if quiz_state['options'].index(selected_option) == quiz_state['correct_option_index']:
                st.success("Correct!")
                st.session_state.quiz_score["correct"] += 1
            else:
                st.error("Incorrect. Try again.")
            st.session_state.quiz_state = None  # Reset quiz state after checking the answer
        if st.session_state.quiz_score['total']==0:
            pass
        else:
            st.write(f"**Score:** {st.session_state.quiz_score['correct']} out of {st.session_state.quiz_score['total']} "
                 f"({(st.session_state.quiz_score['correct'] / st.session_state.quiz_score['total']) * 100:.2f}%)")

        if st.button("Next Question"):
            st.session_state.quiz_state = None  # Reset quiz state to generate a new question

with s2:
    st.header("Images")
    if st.session_state.history:
        q,r = st.session_state.history[-1]
        google_Crawler.crawl(keyword = f'show relevant Diagram or picture from NCERT textbook - Question: {q}, Answer: {r}', max_num = 5)
        image_folder = 'Images'
        if os.path.exists(image_folder):
            images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]
            if images:
                with st.container(border=True, height=400):
                    st.image(images, caption=[os.path.basename(img) for img in images], use_container_width=True)
            for img in images:
                os.remove(img)

#---------------------------------------------------------------------------------------------------------------------------------------
