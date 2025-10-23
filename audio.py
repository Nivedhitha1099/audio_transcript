import streamlit as st
import whisper
import tempfile
import json
import pandas as pd
from datetime import timedelta

st.set_page_config(
    page_title="Audio Transcriber with Word Timestamps",
    layout="wide",
    page_icon="🎙️"
)

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("large-v3")  
    return model

# Function to format time as HH:MM:SS
def format_time(seconds):
    return str(timedelta(seconds=round(seconds, 2)))

def display_transcription_table(transcript):
    rows = []
    for segment in transcript['segments']:
        for word in segment['words']:
            rows.append({
                "Word": word['word'],
                "Start Time": format_time(word['start']),
                "End Time": format_time(word['end'])
            })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

def main():
    st.title("🎧 AI Audio to Text Transcriber with Word Timestamps")
    st.markdown("Upload your audio file (.mp3, .wav, .m4a, etc.) to transcribe with precise timing for every word.")

    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name

        st.audio(uploaded_file, format="audio/wav")

        model = load_whisper_model()
        with st.spinner("Transcribing... this may take a moment."):
            result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

        # Show full text
        st.subheader("Transcribed Text")
        st.write(result["text"])

        # Show word-level timestamps
        st.subheader("Word-Level Timestamps")
        display_transcription_table(result)

        # Export as JSON
        json_result = json.dumps(result, indent=2)
        st.download_button(
            label="Download JSON Transcript",
            data=json_result,
            file_name="word_timestamps.json",
            mime="application/json"
        )

    st.markdown("---")
    

if __name__ == "__main__":
    main()
