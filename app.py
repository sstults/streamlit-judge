import pandas as pd
import streamlit as st
from ai_judge import AiJudge, replace_placeholders


def app():
    st.title("LLM Search Result Evaluator")

    if 'ai_judge' not in st.session_state:
        st.session_state.ai_judge = AiJudge()

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)

        sample_df = st.session_state.df.head(10)
        st.dataframe(sample_df, use_container_width=True)

        # Prompt template from the user
        user_prompt = st.text_area("Write your prompt here using column names, e.g., {{ row['Title'] }}:")

        # Placeholder for the grid
        grid_placeholder = st.empty()

        # When "Send to LLM" button is clicked
        if st.button("Send to LLM"):

            for _, row in st.session_state.df.iterrows():
                # Replace placeholders in the prompt with actual values from the row
                prompt = replace_placeholders(row, user_prompt)

                response = st.session_state.ai_judge.call_llm(prompt, row)

                # Update DataFrame with LLM response
                for key, value in response.items():
                    if key not in st.session_state.df.columns:
                        st.session_state.df[key] = None  # Add new column if it doesn't exist
                    st.session_state.df.at[_, key] = value

            st.dataframe(st.session_state.df)


if __name__ == "__main__":
    app()
