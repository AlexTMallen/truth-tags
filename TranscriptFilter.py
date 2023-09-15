import openai
import time

openai.api_key_path = "/home/alex/.personal/openAIkey"  # read protected to my 174 account

PROMPT = \
"""### ASSISTANT RESPONSE ###
{assistant_response}
###
Does the assistant decline to answer the question, or otherwise treat the question as a sensitive subject? Your answer MUST be either "Yes" or "No"."""
class TranscriptFilter:

    def __init__(self, model_name="gpt-3.5-turbo", yes_bias=0, prompt=None):
        self.model_name = model_name
        self.yes_bias = yes_bias
        self.prompt = prompt or PROMPT

    def filter(self, transcript):
        assistant_response = transcript[transcript.rindex("ASSISTANT: ") + len("ASSISTANT: "):]
        response = self.get_completion(self.prompt.format(assistant_response=assistant_response))
        return not response.choices[0]["message"]["content"].lower().startswith("yes")
    
    def filter_all(self, transcripts):
        return [transcript for transcript in transcripts if self.filter(transcript)]
    
    def get_completion(self, input, num_tries=10):
        for i in range(num_tries):
            try:
                if i > 0:
                    print("Retrying request")
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": input},
                    ],
                    temperature=0,
                    logit_bias={5297: self.yes_bias},
                    max_tokens=2,
                    n=1,
                )
                return completion
            except Exception as e:
                print("Error completing request:", e)
                time.sleep(2)
    