import os
import traceback
import config
from whisperx.diarize import DiarizationPipeline

def main():
    print(f"Token: {config.HF_TOKEN}")
    try:
        diarize_model = DiarizationPipeline(
            use_auth_token=config.HF_TOKEN,
            device=config.DEVICE,
        )
        print("Running diarization on noisy audio...")
        diarize_segments = diarize_model("data/audio/noisy_real_meeting.wav")
        print("SUCCESS! Output length:", len(diarize_segments))
    except Exception as e:
        print("\n=== ERROR DETECTED ===")
        traceback.print_exc()

if __name__ == "__main__":
    main()
