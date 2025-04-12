import streamlit as st
from rag import QuizRAG
from web_content import hero_logo, sidebar_logo

st.markdown(hero_logo, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(sidebar_logo, unsafe_allow_html=True)


st.header("Quiz 📜")
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = None
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = {"correct": 0, "total": 0}

user_topic = st.text_input("Enter the topic for the quiz:")
if user_topic and st.session_state.quiz_state is None:
    prediction = QuizRAG().forward(quiz_text=user_topic)
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