import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

def run_whisper_vaani():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "openai/whisper-small" # Whisper small supports Hindi
    
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

    ds = load_dataset("ARTPARK-IISc/Vaani", "Hindi", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    for i, sample in enumerate(ds.take(6)):
        audio_data = sample["audio"]["array"]
        # Whisper requires input_features
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(inputs)

        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"[Whisper Sample {i+1}] AI Said: {prediction}")

if __name__ == "__main__":
    run_whisper_vaani()
