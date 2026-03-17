import streamlit as st
from rag import QuizRAG
from web_content import hero_logo, sidebar_logo

st.markdown(hero_logo, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(sidebar_logo, unsafe_allow_html=True)

st.header("Quiz 📜")

# ---------------------------
# Session State Init
# ---------------------------
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None

if 'current_q' not in st.session_state:
    st.session_state.current_q = 0

if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = {"correct": 0, "total": 0}

# ---------------------------
# Input Topic
# ---------------------------
user_topic = st.text_input("Enter the topic for the quiz:")

# ---------------------------
# Generate Quiz (JSON)
# ---------------------------
if user_topic and st.session_state.quiz_data is None:
    with st.spinner("Generating quiz..."):
        result = QuizRAG().generate(user_topic)  # ✅ changed

        st.session_state.quiz_data = result
        st.session_state.current_q = 0
        st.session_state.quiz_score = {"correct": 0, "total": 0}

# ---------------------------
# Display Quiz
# ---------------------------
if st.session_state.quiz_data:
    quiz_data = st.session_state.quiz_data
    q_index = st.session_state.current_q

    questions = quiz_data["questions"]

    if q_index < len(questions):
        q = questions[q_index]

        st.write(f"**Topic:** {quiz_data['topic']}")
        st.write(f"**Question {q_index + 1}:** {q['question']}")

        selected_option = st.radio(
            "Select an option:",
            q["options"],
            key=f"quiz_option_{q_index}"
        )

        if st.button("Check Answer"):
            st.session_state.quiz_score["total"] += 1

            if selected_option == q["answer"]:
                st.success("Correct!")
                st.session_state.quiz_score["correct"] += 1
            else:
                st.error(f"Incorrect. Correct answer: {q['answer']}")

        # Score
        if st.session_state.quiz_score["total"] > 0:
            score = st.session_state.quiz_score
            st.write(
                f"**Score:** {score['correct']} / {score['total']} "
                f"({(score['correct'] / score['total']) * 100:.2f}%)"
            )

        # Next Question
        if st.button("Next Question"):
            st.session_state.current_q += 1

    else:
        st.success("🎉 Quiz Completed!")

        score = st.session_state.quiz_score
        st.write(
            f"**Final Score:** {score['correct']} / {score['total']} "
            f"({(score['correct'] / score['total']) * 100:.2f}%)"
        )

        if st.button("Restart Quiz"):
            st.session_state.quiz_data = None
            st.session_state.current_q = 0
            st.session_state.quiz_score = {"correct": 0, "total": 0}