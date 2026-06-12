"""CLI entrypoint:
    python main.py chat     # interactive terminal chatbot (default)
    python main.py serve    # FastAPI server at :8000
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if cmd == "chat":
        from app.graph import ask
        print("Hospital chatbot — type 'exit' to quit.")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            try:
                print("Bot:", ask(q))
            except Exception as e:
                print(f"[error] {e}")

    elif cmd == "serve":
        import os
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("server:app", host="0.0.0.0", port=port)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
