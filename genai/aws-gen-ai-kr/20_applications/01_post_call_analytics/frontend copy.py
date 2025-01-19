import streamlit as st
import boto3
import json
import time
from urllib.request import urlopen
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


st.title("Post Call Analytics Using Amazon Bedrock")

# Setup AWS clients
region = st.sidebar.text_input("AWS Region", "us-west-2")
s3_client = boto3.client('s3', region_name=region)
transcribe_client = boto3.client('transcribe', region_name=region)
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

# S3 bucket setup
#prefix = st.sidebar.text_input("S3 Bucket Prefix", "your-prefix")
prefix = "hsw"
bucket_name = f'bedrock-training-{prefix}'

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
if uploaded_file is not None:
    # Save the file to S3
    s3_path = f's3://{bucket_name}/records/{uploaded_file.name}'
    s3_client.upload_fileobj(uploaded_file, bucket_name, f"records/{uploaded_file.name}")
    st.success(f"File uploaded to {s3_path}")

    # Transcribe the audio
    if st.button("Transcribe Audio"):
        job_name = f'{prefix}-stt-job-{int(time.time())}'
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_path},
            MediaFormat='wav',
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
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            response = urlopen(transcript_uri)
            data = json.loads(response.read())
            transcript = data['results']['transcripts'][0]['transcript']
            st.success("Transcription completed")
            st.text_area("Transcript", transcript, height=200)

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

# Analysis functions
def run_analysis(transcript, question, template):
    prompt = PromptTemplate(template=template, input_variables=["transcript", "question"])
    analysis_prompt = prompt.format(transcript=transcript, question=question)
    response = llm(analysis_prompt)
    return response

# Analysis section
if 'transcript' in locals():
    st.header("Post Call Analysis")
    analysis_type = st.selectbox("Select Analysis Type", ["Summary", "Sentiment", "Intent", "Resolution"])
    
    if analysis_type == "Summary":
        if st.button("Generate Summary"):
            summary = run_analysis(transcript, "Summarize the conversation", summary_template_ko)
            st.write(summary)
    elif analysis_type in ["Sentiment", "Intent", "Resolution"]:
        question = st.text_input(f"Enter {analysis_type} question")
        if st.button(f"Analyze {analysis_type}"):
            result = run_analysis(transcript, question, question_template_ko)
            st.write(result)

st.sidebar.info("This app demonstrates post-call analytics using Amazon Bedrock and Streamlit.")
