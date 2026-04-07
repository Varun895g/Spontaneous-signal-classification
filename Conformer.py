import torch
from transformers import ConformerForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio

def run_conformer_vaani():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "facebook/conformer-sha-base" 
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = ConformerForCTC.from_pretrained(MODEL_ID).to(device)

    ds = load_dataset("ARTPARK-IISc/Vaani", "Hindi", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    for i, sample in enumerate(ds.take(6)):
        audio_data = sample["audio"]["array"]
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            logits = model(inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0]
        print(f"[Conformer Sample {i+1}] AI Said: {prediction}")

if __name__ == "__main__":
    run_conformer_vaani()
