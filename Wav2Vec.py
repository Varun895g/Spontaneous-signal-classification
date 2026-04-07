import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio

# CONFIGURATION
HF_TOKEN = "hf_your_token_here"  # Paste your token here
MODEL_ID = "ai4bharat/indicwav2vec-hindi" 
LANGUAGE_SUBSET = "Hindi" # Options: "Hindi", "Bhojpuri", "Bengali", etc.
NUM_TEST_SAMPLES = 6

def run_vaani_mini_test():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Model & Processor (Once)
    print(f"Loading model: {MODEL_ID}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)

    # 3. Stream VAANI Dataset
    # streaming=True is key: it fetches metadata without downloading the audio files
    print(f"Connecting to VAANI ({LANGUAGE_SUBSET})...")
    ds = load_dataset(
        "ARTPARK-IISc/Vaani", 
        LANGUAGE_SUBSET, 
        split="train", 
        streaming=True, 
        token=HF_TOKEN
    )
    
    # 4. Auto-resample to 16kHz (Required for Wav2Vec2)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # 5. Run Inference on exact number of samples
    print(f"\n--- Testing {NUM_TEST_SAMPLES} Samples ---")
    
    # .take(n) stops the stream after fetching n items
    for i, sample in enumerate(ds.take(NUM_TEST_SAMPLES)):
        audio_data = sample["audio"]["array"]
        ground_truth = sample["transcription"]
        
        # Preprocess
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_values.to(device)

        # Inference
        with torch.no_grad():
            logits = model(inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0]

        print(f"\n[SAMPLE {i+1}]")
        print(f"Expected: {ground_truth}")
        print(f"AI Said : {prediction.lower()}")
        print("-" * 30)

if __name__ == "__main__":
    run_vaani_mini_test()
