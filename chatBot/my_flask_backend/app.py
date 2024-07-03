import os
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Define the model to be downloaded
model_id = 'TheBloke/Orca-2-13B-GGUF'

# Define the location of Google Drive to download the LLM model locally
os.environ['XDG_CACHE_HOME'] = 'drive/MyDrive/LLM_data/model/cache/'
config = {'temperature': 0.00, 'max_new_tokens': 512, 'context_length': 4000, 'gpu_layers': 50, 'repetition_penalty': 1}

# Download the LLM model locally
llm = CTransformers(model=model_id,
                    HF_TOKEN='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',
                    model_type="llama",
                    gpu_layers=50,
                    device=0,
                    config=config,
                    callbacks=[StreamingStdOutCallbackHandler()])

# Predefined services and descriptions
services = {
    "Web Development": "Provides services for building and maintaining websites.",
    "Graphic Design": "Offers design services for creating visual content.",
    "Digital Marketing": "Specializes in online advertising and promotions."
}

def generate_service_recommendation(user_input, services):
    prompt = f"User needs help with: {user_input}\nBased on their needs, the best services we offer are:\n"
    for service, description in services.items():
        prompt += f"- {service}: {description}\n"
    return llm(prompt)

def main():
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_service_recommendation(user_input, services)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
