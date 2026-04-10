import traceback
import config
from whisperx.diarize import DiarizationPipeline
import sys

def main():
    try:
        print("Instantiating Model...", file=sys.stderr)
        diarize_model = DiarizationPipeline(
            token=config.HF_TOKEN,
            device=config.DEVICE,
        )
        print("Model Loaded. Diarizing...", file=sys.stderr)
        diarize_segments = diarize_model("data/audio/noisy_real_meeting.wav")
        print("SUCCESS! Segments:", len(diarize_segments), file=sys.stderr)
    except Exception as e:
        print("\n=== TRACEBACK ===", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()
