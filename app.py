import streamlit as st

#For Sub-Process
import os
import subprocess

from agents import fetch_raw_content, llm, generate_questions, evaluate_answers

#Env path for CV
cv_env_path = r"C:\Users\SHIVAM\PycharmProjects\Face_rec\cv_env"

#Script path of Executable
cv_script_path = r"C:\Users\SHIVAM\Desktop\MVP\GazeTracking\gaze_tracking\gaze_tracking.py"

#To activate env
def activate_venv(env_path):
    activate_script = os.path.join(env_path, 'Scripts', 'activate.bat')
    return f'call "{activate_script}" && '

#here for runnning the scripts
def run_script(script_path, env_path):
    python_path = os.path.join(env_path, 'Scripts', 'python.exe')
    command = f'{activate_venv(env_path)} "{python_path}" "{script_path}"'
    subprocess.run(command, shell=True)
    
    
    
def main():
    st.title("AI Agentt MVP")
    
    if 'stage' not in st.session_state:
        st.session_state.stage = 'input'

    if st.session_state.stage == 'input':
        repo_url = st.text_input("Enter the GitHub repository URL:")
        file_path = st.text_input("Enter the main file path:")

        if st.button("Fetch Raw Content and Generate Questions"):
            raw_content = fetch_raw_content(repo_url, file_path)
            if raw_content:
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_questions(raw_content)
                st.session_state.stage = 'answer'
                st.experimental_rerun()

    elif st.session_state.stage == 'answer':
        st.subheader("Interview Questions and Answers")
        
        questions = st.session_state.questions.split('\n')
        st.session_state.answers = {}

        for i, question in enumerate(questions, 1):
            if question.strip():
                st.write(f"Q{i}: {question}")
                answer = st.text_area(f"Your Answer for Q{i}", key=f"answer_{i}", height=100)
                st.session_state.answers[f'Answer_{i}'] = answer

        if st.button("Submit All Answers"):
            if all(st.session_state.answers.values()):
                st.session_state.stage = 'evaluate'
                st.experimental_rerun()
            else:
                st.warning("Please answer all questions before submitting.")

    elif st.session_state.stage == 'evaluate':
        st.subheader("Evaluation")
        with st.spinner("Evaluating answers..."):
            evaluation = evaluate_answers(st.session_state.questions, st.session_state.answers)
        st.write(evaluation)

        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

if __name__ == "__main__":
    main()