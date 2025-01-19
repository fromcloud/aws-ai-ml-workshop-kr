import streamlit as st
import boto3
import json
import time
from urllib.request import urlopen
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import IPython.display as ipd


st.title("Bedrock 을 이용한 보이스 콜 분석")
#st.set_page_config(initial_sidebar_state="auto")
st.sidebar.info("This app demonstrates post-call analytics using Amazon Bedrock and Streamlit.")


# Setup AWS clients
region = st.sidebar.text_input("AWS Region", "us-west-2")
s3_client = boto3.client('s3', region_name=region)
transcribe_client = boto3.client('transcribe', region_name=region)
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

# template definition
summary_template_ko = """\n\nHuman:
아래의 리테일 지원 통화 기록을 분석하세요. 전체 문장으로 대화에 대한 자세한 요약을 제공하세요.

통화: "{transcript}"

요약:

\n\nAssistant:"""

question_template_ko = """\n\nHuman:

아래의 통화 기록을 바탕으로 질문에 답하세요.
"<시작시간>" 이어지는 문장의 시작시간을 나타내고 "<종료시간>"은 앞 문장의 종료시간을 나타냅니다.

통화: "{transcript}"

질문: "{question}"

응답:

\n\nAssistant:"""

question_time_template_ko = """\n\nHuman:

아래의 통화 기록을 바탕으로 질문에 답하세요.
"<시작시간>" 이어지는 문장의 시작시간을 나타내고 "<종료시간>"은 앞 문장의 종료시간을 나타냅니다.

통화: "{transcript}"

질문: "{question} 답변과 함께 답변을 위해 참조한 대화의 시작 및 종료시간을 아래 형태로 알려주세요. \n답변:\n시작시간:\n종료시간:"

응답:

\n\nAssistant:"""


# S3 bucket setup
#prefix = st.sidebar.text_input("S3 Bucket Prefix", "your-prefix")
prefix = "hsw"
bucket_name = f'bedrock-training-{prefix}'

# LLM setup
llm = Bedrock(
    #model_id="Claude-V2-1",
    model_id="anthropic.claude-v2",
    client=bedrock_client,
    model_kwargs={
        "max_tokens_to_sample": 512,
        "temperature": 0,
        "top_p": 0.999,
    },
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

PARAMS = {
    "max_tokens_to_sample":512,
    "stop_sequences":["\n\nHuman", "\n\n인간", "\n\n상담사", "\n\n\n", "\n\n질문"],
    "temperature":0,
    "top_p":0.999
}

#@st.cache_data
def analysis(_llm, transcript, params, question=None, template="", max_tokens=50):
    if question is None:
        prompt = PromptTemplate(template=template, input_variables=["transcript"])
        analysis_prompt = prompt.format(transcript=transcript)
    else:
        prompt = PromptTemplate(template=template, input_variables=["transcript", "question"])
        analysis_prompt = prompt.format(
            transcript=transcript,
            question=question
        )
    _llm.model_kwargs = params
    response = _llm(analysis_prompt)
    return response


@st.cache_data
def spk_sperator(data):
    previos_spk = ""
    contents, contents_temp = [], []
    end_time = None

    for res in data["results"]["items"]:
        speaker_label = res["speaker_label"]
        content = res["alternatives"][0]["content"]
        start_time = res.get("start_time", None)

        if previos_spk != speaker_label:
            contents_temp.append(f'<종료시간:{end_time}>')
            contents.append(" ".join(contents_temp))
            contents_temp = []

            contents_temp.append(f'{speaker_label}:<시작시간:{start_time}>')
            contents_temp.append(content)
        else:
            contents_temp.append(content)
            if content not in ["?", ",", "."]: end_time = res.get("end_time", None)

        previos_spk = speaker_label

    contents_temp.append(f'<종료시간:{end_time}>')
    contents.append(" ".join(contents_temp))

    return "\n".join(contents[1:])




@st.cache_data
def upload_file(file):
    text = ""
    if file is not None:
        # Save the file to S3
        st.write(file)
        s3_path = f's3://{bucket_name}/records/{uploaded_file.name}'
        s3_client.upload_fileobj(uploaded_file, bucket_name, f"records/{uploaded_file.name}")
        st.success(f"File uploaded to {s3_path}")
        st.audio("./records/voice-examples.wav", format="audio/wav", loop=True)
        #st.audio(file.name, format="audio/wav", loop=True)
        return s3_path



@st.cache_data
def transcribe(transcribe_button, s3_path):
    # Transcribe the audio
    #if transcribe_button:
    job_name = f'{prefix}-stt-job-{int(time.time())}'
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_path},
        MediaFormat='wav',
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 2,
        },
        LanguageCode='ko-KR'
    )
    st.info("Transcription job started. Please wait...")

    # Wait for transcription to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)
    
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        st.success("Transcription completed")
        response = urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        data = json.loads(response.read())
        transcribed_text = data['results']['transcripts'][0]['transcript']
        separated_text = spk_sperator(data)
        return transcribed_text, separated_text
            


@st.cache_data
def summary(text):    
    summary = analysis(
        _llm=llm,
        transcript=text,
        params=PARAMS,
        template=summary_template_ko
    )
    return summary


@st.cache_data
def response(question, text):
    res = analysis(
        _llm=llm,
        transcript=text,
        params=PARAMS,
        question=question,
        template=question_template_ko
    )
    return res
    #if not llm.streaming: print (res)


uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
if "s3_path" not in st.session_state:
    s3_path = ""

if upload_file:
    s3_path = upload_file(uploaded_file)
    st.session_state.s3_path = s3_path


if "transcribed_text" not in st.session_state:
    transcribed_text = ""

if "separated_text" not in st.session_state:
    separated_text = ""

transcribe_button = st.button("Transcribe Audio")
if transcribe_button and s3_path:
    transcribed_text, separated_text  = transcribe(transcribe_button, s3_path)
    st.text_area("Transcription", transcribed_text, height=200)
    st.text_area("Separated transcription ", separated_text, height=200)
    st.session_state.transcribed_text = transcribed_text
    st.session_state.separated_text = separated_text
    #st.write("transcribed_text :", transcribed_text)


if "summary_text" not in st.session_state:
    summary_text = ""

summary_button = st.button("Summary")

if summary_button:
    summary_text = summary(st.session_state.transcribed_text)
    st.text_area("summary", summary_text, height=200)
    st.session_state.summary_text = summary_text




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("이 콜에 대해 물어보세요 ...")


if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Simulate bot response
    time.sleep(1)  # Simulate processing time
    bot_response = f"You said: {prompt}"

    res = response(prompt, st.session_state.transcribed_text)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": res})
    
    # Display bot response
    with st.chat_message("assistant"):
        st.write(res)

