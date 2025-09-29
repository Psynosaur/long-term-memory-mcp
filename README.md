# Robust Long‑Term Memory MCP for LM Studio  
  
This project implements a **persistent memory system** for AI companions running in [LM Studio](https://lmstudio.ai), powered by a hybrid of **SQLite (structured storage)** and **ChromaDB (vector search)**. It is designed for decades‑long use, seamless recall across sessions, and automatic backups — making your AI companion feel like a continuous, living persona.  
  
---  
  
## ✨ Features  
  
- **Hybrid Memory System**  
  - **SQLite** for structured metadata and fast queries    
  - **ChromaDB** for semantic similarity and natural recall    
  - **JSON backups** for portability    
  
- **Cross‑chat continuity**: memories persist beyond a single chat    
- **Cross‑model continuity**: swap models freely, the memory stays intact    
- **Cross‑machine portability**: move the database to another system and continue seamlessly    
- **Automatic backups**: daily backups and after every 100 memories, pruned to keep the last 10    
- **Invisible memory integration**: tools are hidden from the user; conversations feel natural    
  
---  
  
## 📦 Installation  
  
1. Clone the repo:  
   ```bash  
   git clone https://github.com/Rotoslider/long-term-memory-mcp.git
     
   cd long-term-memory-mcp  


2. Install requirements:

```
    pip install -r requirements.txt
```
Requirements include:

    chromadb
    sentence-transformers
    sqlite3 (built into Python)
    fastmcp

3. (Optional) For faster HuggingFace model fetching:

```
    pip install "huggingface_hub[hf_xet]"  
```
🚀 Running the Memory MCP

edit the LMStudio mcp.json file to include the correct path:

```
    {
  "mcpServers": {
    "long_term_memory": {
      "command": "C:\\Python313\\python.exe",
      "args": [
        "D:\\a.i. apps\\long_term_memory_mcp\\LongTermMemoryMCP.py"
      ],
      "env": {}
    }
  }
}  
```
Then, in LM Studio:

    Go to Server (MCP) Settings
    Load the MCP: "long_term_memory"

🧠 How Memory Works

    Cross‑Chats → Start a new chat — memories are still there.
    Cross‑Models → Switch models — the same memory remains available.
    Cross‑Machines → Copy the database folder (memory_db/ and memory_backups/) and your system prompt, point to the path, and everything carries over.

💡 Think of it as your AI’s diary: chats are conversations, the database is her journal.

## Environment variable for custom data dir:


### Windows PowerShell
```
$env:AI_COMPANION_DATA_DIR="D:\a.i. apps\long_term_memory_mcp\data"  
```  
### Linux/macOS
```
export AI_COMPANION_DATA_DIR="/home/username/ai_companion_data"  
```
📂 Backups

Backups are created automatically:

    Every 24 hours
    Or after 100 new memories
    Stored in memory_backups/ with timestamped folders
    Only the last 10 backups are kept automatically

Each backup includes:

    SQLite DB copy
    ChromaDB copy
    JSON export of all memories (portable and future‑proof)

📝 Recommended Setup in LM Studio

For a natural, invisible memory experience, add a system prompt that guides the model to:

    Store new facts in memory with remember
    Recall them naturally with search_memories
    Never expose tool calls to the user

Example prompt:

    “You are an AI companion with long‑term memory. Store facts naturally (‘Got it, I’ll remember that’). Recall them when asked in natural language. Never expose internal tool usage to the user.”

🛠 Contributing

Pull requests welcome!

    Found a bug? Open an issue.
    Want to add features (custom backup schedule, encryption, etc.)? Let’s collaborate.

📜 License

MIT

🔥 With this setup, your AI can build a persistent, evolving memory that feels natural across conversations, models, and even years.
