import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

#path
adapter_path = "./manim_cli_2"  

# get base model
peft_config = PeftConfig.from_pretrained(adapter_path)

# base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

# fine tuned model
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print(adapter_path)
print("Enter your prompt for code generation (type 'exit' to quit):")
while True:
    prompt = input(">>> ")
    if prompt.strip().lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        do_sample=True,
        temperature=0.4,
        top_p=0.9
    )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Generated Code ---\n")
    print(generated_code)
    print("\n----------------------\n")
