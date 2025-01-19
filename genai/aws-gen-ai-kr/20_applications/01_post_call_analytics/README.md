## 동작방식
- wav file upload to s3
- Amazon Transcibe 를 이용한 STT (화자인식)
- Amazon Bedrock 을 이용한 텍스트 요약 
- Amazon Bedrock 을 이용한 Q&A chatting

## 실행방법
```streamlit run frontend.py```


## Sample Questions
- 고객의 감정은 어떤가요?
- 문제에 대한 개선을 위해서 어떤 방법이 있을까요?
- 학습지는 언제 종료되나요?
- 환불은 언제 가능한가요?
- 결제된 금액은 얼마인가요?
- 상담원의 이름은 무엇인가요?
- 상담을 요청한 사람의 성별은 무엇입니까?
- 상담을 요청한 사람의 나이는 어떻게 되나요?