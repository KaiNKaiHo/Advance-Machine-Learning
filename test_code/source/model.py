from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
def create_model(model_id= "Qwen/Qwen3-VL-4B-Instruct"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,)
    return model, processor
def generate_text_from_sample(model, processor, sample, is_labeled, max_new_tokens=1000, device="cuda"):
    if is_labeled == True:

        # Prepare the text input by applying the chat template
        text_input = processor.apply_chat_template(
            sample[5]['messages'][:2],  # Use the sample without the system message
            tokenize=False,
            add_generation_prompt=True
        )

        # Process the visual input from the sample
        image_inputs, _ = process_vision_info(sample[5]['messages'])
    
    else:
        # Prepare the text input by applying the chat template
        text_input = processor.apply_chat_template(
            sample[3]['messages'][:2],  # Use the sample without the system message
            tokenize=False,
            add_generation_prompt=True
        )

        # Process the visual input from the sample
        image_inputs, _ = process_vision_info(sample[3]['messages'])        
    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    print(generated_ids)
    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(output_text)
    print(output_text[0])
    # return output_text[0]  # Return the first decoded output text
    return output_text[0]