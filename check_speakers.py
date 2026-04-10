import json

d = json.load(open('data/meetings/test_gpu_speakers.json'))
with open('data/speaker_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"PARTICIPANTS: {d['meeting']['participants']}\n")
    f.write(f"TOTAL CHUNKS: {d['meeting']['chunk_count']}\n\n")
    for c in d['chunks']:
        f.write(f"Chunk {c['chunk_id']}: speakers={c['speakers']}\n")
        f.write(f"  text: {c['text'][:120]}...\n\n")
print("Done! Check data/speaker_results.txt")
