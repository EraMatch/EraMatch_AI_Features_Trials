import ollama
import sys

# Define the host explicitly in case it's not localhost default, 
# although user error showed 127.0.0.1:11434. 
# We'll rely on the default client which uses OLLAMA_HOST env var or default.
host = "http://127.0.0.1:11434"
client = ollama.Client(host=host)

models = [
    "qwen2.5-coder:14b",
    "llama3.3",         # Warning: This is 70B parameters
    "deepseek-r1:14b",
    "qwen2.5-coder:32b"
]

print(f"🔌 Connecting to Ollama at {host}...")
print("📉 Starting download of heavy models. This may take significant time/bandwidth...")

for model in models:
    print(f"\n⬇️ Pulling {model}...")
    try:
        # Stream pull to keep connection alive
        for progress in client.pull(model, stream=True):
             status = progress.get('status', '')
             completed = progress.get('completed', 0)
             total = progress.get('total', 1)
             if total > 0 and 'downloading' in status.lower():
                 percent = (completed / total) * 100
                 if int(percent) % 10 == 0: # Reduce spam
                     sys.stdout.write(f"\r  {status}: {percent:.1f}%")
                     sys.stdout.flush()
        print(f"\n✅ {model} Pulled Successfully!")
    except Exception as e:
        print(f"\n❌ Failed to pull {model}: {e}")
        
print("\n🎉 All operations completed.")
