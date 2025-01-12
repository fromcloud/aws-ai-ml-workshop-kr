import streamlit as st
import dotenv
import boto3
import io

import os
import random
import json

dotenv.load_dotenv()

st.set_page_config(layout="wide")
# Reference: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
st.title('교육 기업 VoC Call 분석을 위한 채팅')
st.subheader("Powered by Amazon Berock with Anthropic Claude v2",divider="rainbow")



bucket_name = 'bedrock-training-hsw'
s3 = boto3.client('s3')
uploaded_file = st.file_uploader("VoC 음성 wav 파일을 선택하세요", type='wav')
if uploaded_file is not None:
    # Display file details
    #st.write("Filename:", uploaded_file.name)
    #st.write("File type:", uploaded_file.type)
    #st.write("File size:", uploaded_file.size, "bytes")

    # Upload button
    if st.button("Upload to S3"):
        try:
            # Upload file to S3
            s3.upload_fileobj(
                uploaded_file,
                bucket_name,
                uploaded_file.name
            )
            st.success(f"File {uploaded_file.name} successfully uploaded to S3!")
        except Exception as e:
            st.error(f"Error uploading file to S3: {str(e)}")


# Play button
if st.button("Play"):
  # Create a temporary file-like object
  audio_bytes = io.BytesIO(uploaded_file.read())

  # Play the WAV file
  st.audio(audio_bytes, format='audio/wav')



@st.cache_data
def get_welcome_message() -> str:
    return random.choice(
        [
            "안녕하세요. VoC 콜 분석 채팅 앱 데모입니다.",
            "voice file 을 업로드하고 채팅을 시작해보시기 바랍니다.",
            "이 데모에서는 wav 파일을 지원합니다. 그러나 필요에 따라 다양한 음성 파일을 사용하실 수 있습니다.",
        ]
    )


@st.cache_resource
def get_bedrock_client():
    return boto3.client(service_name='bedrock-runtime')


def get_history() -> str:
    hisotry_list = [
        f"{record['role']}: {record['content']}" for record in st.session_state.messages
    ]
    return '\n\n'.join(hisotry_list)


client = get_bedrock_client()
modelId = 'anthropic.claude-v2:1'
accept = 'application/json'
contentType = 'application/json'


welcome_message = get_welcome_message()
with st.chat_message('assistant'):
    st.markdown(welcome_message)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    display_role = 'user'
    if message['role'] == 'Assistant':
        display_role = 'assistant'

    with st.chat_message(display_role):
        st.markdown(message["content"])


def parse_stream(stream):
    full_response = ""
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            message = json.loads(chunk.get('bytes').decode())[
                'completion'] or ""
            full_response += message
            yield message
    st.session_state.messages.append(
        {"role": "Assistant", "content": full_response}
    )


if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "Human", "content": prompt})
    with st.chat_message("Human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        history = get_history()
        body = json.dumps({
            "prompt": f"{history}\n\nAssistant:",
            "max_tokens_to_sample": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
        })
        response = client.invoke_model_with_response_stream(
            body=body,
            modelId=modelId,
        )
        stream = response.get('body')
        if stream:
            # st.write_stream is introduced in streamlit v1.31.0
            st.write_stream(parse_stream(stream))


if DEBUG := os.getenv("DEBUG", False):
    st.subheader("History", divider="rainbow")
    history_list = [
        f"{record['role']}: {record['content']}" for record in st.session_state.messages
    ]
    st.write(history_list)