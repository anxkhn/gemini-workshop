# Contributing

This is a workshop project designed for learning. Contributions that improve clarity and fix bugs are welcome!

## Beginner Workflow

1. **Fork** this repo on GitHub.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gemini-workshop.git
   cd gemini-workshop
   ```
3. **Create a branch**:
   ```bash
   git checkout -b my-fix
   ```
4. **Set up the environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. **Make your changes**. The key files are:
   - `backend/config.py` - Configuration and defaults
   - `backend/main.py` - FastAPI endpoints
   - `backend/rag.py` - RAG pipeline
   - `backend/tools.py` - Tool definitions
   - `static/index.html` - The UI
6. **Run tests**:
   ```bash
   ./test.sh
   ```
7. **Commit and push**:
   ```bash
   git add -A
   git commit -m "Describe your change"
   git push origin my-fix
   ```
8. **Open a Pull Request** on GitHub.

## Guidelines

- Keep code simple and readable - this is for beginners.
- No external build tools (no npm, no webpack, no Tailwind).
- Test your changes before submitting.
- One file per concern: config, RAG, tools, UI.
