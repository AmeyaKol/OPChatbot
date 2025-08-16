Title: Build a Streamlit AI Document Assistant with three panes (Phase 1 + Phase 2)

Goal
- Build a modern, minimal Streamlit app that feels like a document workspace.
- Three panes: 
  1) Chat pane (left) 
  2) Document pane (center, editable Notepad-like) 
  3) Reference pane (right, uploaded .txt files shown as readable previews)
- Users can select text in the Document pane, right‑click “Interact” (Phase 2), and the selected text is sent as context to the chatbot, focusing the chat input.
- Default LLM: Google Gemini. The app must stream responses.
- Black/white/blue theme. Clean, professional look.

Scope: Phase 1 + Phase 2 only (no RAG yet)

Core requirements
1) Layout and UX
   - Use a true three-column layout with adjustable widths (e.g., via sliders or a “Compact / Comfortable” toggle).
   - Collapsible capability per pane (e.g., expand/collapse Document or Reference panes).
   - Sticky mini-headers for each pane with simple icons:
     - Chat: 💬 Chat
     - Document: 📝 Document
     - References: 📁 References
   - The app must remember pane widths and collapsed state across reruns using `st.session_state`.
   - App loads with dark mode by default.

2) Document pane (editable)
   - Use a code-editor-like component that supports selecting text and retrieving the selection.
   - Preferred: `streamlit-code-editor` (CodeMirror-based) or `streamlit-ace`. Choose whichever reliably returns selected text and exposes selection events.
   - Provide a Notepad-like mode: plain text, no syntax highlight, soft wrap on.
   - Show character/word count.
   - Phase 2: 
     - Detect selection; when a selection exists, show a subtle inline toolbar/button “Interact with selection”.
     - If feasible, implement a right-click context menu item “Interact” (using the editor’s hooks or a small Streamlit component). If right-click is not feasible, the inline button is acceptable.

3) Reference pane
   - `st.file_uploader(accept_multiple_files=True, type=["txt"])`.
   - On upload, read the file(s) (UTF-8, ignore errors), store in `st.session_state.references` as a list of dicts `{ "name": str, "content": str }`.
   - Display a scrollable card per file with:
     - File icon and name
     - Preview (first ~2,000 chars, neatly truncated with “… (truncated)”)
     - A “Copy preview” button
   - These files are automatically part of the chat context. No slash commands needed.

4) Chat pane
   - Use Streamlit’s chat primitives: `st.chat_message`, `st.chat_input`.
   - On send, build a context-aware prompt including:
     - Current document text
     - Uploaded reference files’ content
     - If a selection exists (from Phase 2), include it prominently as “User selection”.
   - Stream responses (`st.write_stream`) from Gemini.
   - Store history in `st.session_state.chat_history`:
     - Structure: list of `{ "role": "user"|"assistant", "content": str }`
   - Provide “Clear chat” action.

5) Model integration (Gemini)
   - Use `google-generativeai` with API key from `.env` (`GEMINI_API_KEY`) and model name from `.env` (`GEMINI_MODEL`, default to `gemini-1.5-flash` or latest available streaming-capable model).
   - Implement a small LLM client function:
     - `call_gemini_stream(prompt: str) -> generator[str]`
     - Use `model.generate_content([...], stream=True)` and yield chunks’ text safely.

6) Theme and styling
   - `.streamlit/config.toml` to set dark mode and primary color (blue).
   - Add custom CSS for:
     - Three-column layout spacing and gutters
     - Card styling for reference previews
     - Clean typography and better contrast
   - Palette: black / white / blue similar to the previous theme:
     - Background: near-black
     - Foreground: near-white
     - Primary: blue
     - Borders: subtle neutral grey

7) State model (must use these keys)
   - `st.session_state.document_text: str`
   - `st.session_state.selected_text: str | None`
   - `st.session_state.references: list[{"name": str, "content": str}]`
   - `st.session_state.chat_history: list[{"role": "user"|"assistant", "content": str}]`
   - `st.session_state.pane_widths: {"chat": float, "doc": float, "ref": float}`
   - `st.session_state.collapsed: {"chat": bool, "doc": bool, "ref": bool}`

8) Prompt builder (high-level)
   - Function: `build_prompt(user_message: str, document_text: str, references: list, selected_text: str | None) -> str`
   - Behavior:
     - If `selected_text` exists, show it first as a quoted block and instruct the model to focus on it if relevant.
     - Provide a short system preamble explaining the app and how to use the provided context responsibly (don’t overfit to truncated previews; consider full content).
     - Concise formatting; avoid exceeding token limits (truncate extremely large content intelligently).

9) Performance & safety
   - Truncate only in UI; the full file content remains in memory and is used for prompts.
   - Guardrails: if total context > safe limit, summarize references before building the final prompt (simple heuristic summarization is OK for now).
   - Handle empty states gracefully (no files, empty doc, no selection).
   - Errors should be surfaced in a friendly `st.warning`/`st.error`.

10) Accessibility and polish
   - Keyboard: when “Interact” is used (Phase 2), the selected text is appended to the chat input and focus moves to the chat input.
   - Provide light hover/active effects for buttons and cards.
   - Keep styling minimal and modern.

Minimal file structure to generate
- `app.py` (main Streamlit app)
- `requirements.txt`
- `.env.example` (GEMINI_API_KEY=..., GEMINI_MODEL=gemini-1.5-flash)
- `.streamlit/config.toml` (theme + server config)
- `styles.css` (custom CSS, loaded via `st.markdown(..., unsafe_allow_html=True)`)
- Optional if needed for right-click:
  - `components/context_menu/` with a tiny Streamlit component exposing selection + a custom “Interact” option. If not implemented, use inline “Interact with selection” button above editor.

Functional expectations (Phase 1)
- Three panes render side by side on desktop.
- Uploading .txt files shows readable previews in the right pane.
- Asking a question references both the document and uploaded files as context.
- Gemini streams responses in the chat.

Functional expectations (Phase 2)
- Selecting text in the Document pane enables “Interact with selection”.
- Clicking “Interact” injects the quoted selection into the chat input and focuses it.
- If a right‑click menu is feasible in the chosen editor, wire it to do the same.

Key implementation notes
- Prefer `streamlit-code-editor` (CodeMirror) if it returns `selection` reliably; otherwise use `streamlit-ace`. If neither exposes selection robustly, implement a small custom component or show an inline “Use selection” button when selection length > 0.
- For focusing the chat input, use a small `components.html` with JS to call `document.querySelector('textarea[aria-label="chat input"]').focus()` or similar. Provide graceful fallback if DOM changes.
- For streaming Gemini:
  Pseudocode:
  ```
  import google.generativeai as genai

  def call_gemini_stream(prompt: str):
      model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
      resp = model.generate_content([prompt], stream=True)
      for ev in resp:
          if hasattr(ev, "text") and ev.text:
              yield ev.text
  ```
- Chat history should be rendered using `st.chat_message` for both user and assistant roles.

Acceptance criteria
- The app starts with the three panes visible, with clean dark theme.
- Uploading 1–3 `.txt` files shows each as a card with truncated preview in the Reference pane.
- Typing “Summarize the uploaded document” yields a streamed, coherent answer referencing the uploaded files.
- Selecting a paragraph in the Document pane and hitting “Interact with selection” moves the text into the chat input, focuses it, and the next response reflects that selection in the reasoning.
- Clearing chat resets only the conversation, not the document or references.
- No uncaught exceptions during normal usage.

Windows/PowerShell setup (for the user to run manually)
- Create and activate venv:
  ```
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- Install deps (minimum):
  ```
  pip install streamlit google-generativeai python-dotenv streamlit-code-editor
  ```
  (If `streamlit-code-editor` is unavailable or selection is unreliable, switch to `streamlit-ace` and add it to requirements.)
- Create `.env` with:
  ```
  GEMINI_API_KEY=your_api_key_here
  GEMINI_MODEL=gemini-1.5-flash
  ```
- Run:
  ```
  streamlit run app.py
  ```

Nice-to-haves (don’t block delivery)
- Width slider(s) to adjust column ratios live.
- Light/ Dark mode toggle.
- Copy buttons on previews and model answers.
- Character limit indicator on chat input.

Non-goals (for now)
- RAG (vector stores, chunking, citations).
- Auth/logins.
- Multi-user persistence beyond Streamlit’s default session behavior.