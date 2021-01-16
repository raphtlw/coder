from typing import Any
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer: Any = GPT2Tokenizer.from_pretrained(
    "congcongwang/gpt2_medium_fine_tuned_coder"
)
model: Any = GPT2LMHeadModel.from_pretrained(
    "congcongwang/gpt2_medium_fine_tuned_coder"
)

USE_CUDA = False
LANG = "python"

if USE_CUDA:
    model.to("cuda")


def complete(context: str) -> str:
    input_ids = (
        tokenizer.encode("<python> " + context, return_tensors="pt")
        if LANG == "python"
        else tokenizer.encode("<java> " + context, return_tensors="pt")
    )
    outputs = model.generate(
        input_ids=input_ids.to("cuda") if USE_CUDA else input_ids,
        max_length=128,
        temperature=0.7,
        num_return_sequences=1,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


if __name__ == "__main__":
    while True:
        code = input("Enter some code > ")
        output = complete(code)
        print("Completion:\n")
        print(output)