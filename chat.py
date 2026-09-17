import sys
from inference import MiniLLMInference

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    engine = MiniLLMInference()
    engine.chat()


if __name__ == "__main__":
    main()
