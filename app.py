import os
from typing import Generator, List, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Import Qdrant utilities
try:
    from qdrant_utils import qdrant_manager, search_qdrant, add_document_to_qdrant
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    st.warning("Qdrant utilities not available. Install dependencies with: pip install -r requirements.txt")

# streamlit-code-editor not available, using enhanced textarea fallback
HAS_CODE_EDITOR = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False


# -------------------------
# Session state defaults
# -------------------------
def ensure_state_defaults() -> None:
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "selected_text" not in st.session_state:
        st.session_state.selected_text = None
    if "references" not in st.session_state:
        st.session_state.references = []  # list[{name, content}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[{role, content}]
    if "pane_widths" not in st.session_state:
        st.session_state.pane_widths = {"chat": 0.33, "doc": 0.34, "ref": 0.33}
    if "collapsed" not in st.session_state:
        st.session_state.collapsed = {"chat": False, "doc": False, "ref": False}
    if "file_uploader_trigger" not in st.session_state:
        st.session_state.file_uploader_trigger = False
    # New variables for simple update workflow
    if "pending_document_update" not in st.session_state:
        st.session_state.pending_document_update = None
    if "pending_update_for_selection" not in st.session_state:
        st.session_state.pending_update_for_selection = None
    if "pending_text_replacement" not in st.session_state:
        st.session_state.pending_text_replacement = None
    if "pending_replacement_for_selection" not in st.session_state:
        st.session_state.pending_replacement_for_selection = None
    # New chat mode toggle
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "discuss"  # "discuss" or "update"
    
    # Qdrant integration variables
    if "qdrant_references" not in st.session_state:
        st.session_state.qdrant_references = []  # list[{doc_id, content, score, metadata}]
    if "qdrant_initialized" not in st.session_state:
        st.session_state.qdrant_initialized = False


# -------------------------
# Theming / assets
# -------------------------
def load_styles() -> None:
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


# -------------------------
# Utilities
# -------------------------
SAFE_PROMPT_CHAR_LIMIT = 24000  # rough heuristic before summarization
TRUNCATED_PREVIEW_CHARS = 2000


def count_words_and_chars(text: str) -> Tuple[int, int]:
    cleaned = text.strip()
    if not cleaned:
        return 0, 0
    words = [w for w in cleaned.split() if w]
    return len(words), len(cleaned)


def truncate_for_preview(text: str, max_chars: int = TRUNCATED_PREVIEW_CHARS) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "… (truncated)", True


def naive_summarize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Simple heuristic: keep head and tail
    head = text[: max_chars // 2]
    tail = text[-max_chars // 3 :]
    return head + "\n\n…\n\n" + tail


def gather_context(document_text: str, references: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    doc = document_text or ""
    refs = references or []
    
    # Include Qdrant references if available
    qdrant_refs = st.session_state.get("qdrant_references", [])
    if qdrant_refs:
        # Convert Qdrant results to the same format as manual references
        for qr in qdrant_refs:
            refs.append({
                "name": f"Qdrant: {qr.get('doc_id', 'document')} (Score: {qr.get('score', 0):.3f})",
                "content": qr.get("content", "")
            })
    
    # If no references at all, provide empty reference section
    if not refs:
        refs = [{"name": "No References", "content": "No relevant documents or references found."}]

    total_chars = len(doc) + sum(len(r.get("content", "")) for r in refs)
    if total_chars <= SAFE_PROMPT_CHAR_LIMIT:
        return doc, refs

    # Summarize references first, then document if still too large
    remaining = SAFE_PROMPT_CHAR_LIMIT - len(doc)
    summarized_refs: List[Dict[str, str]] = []
    if remaining < 0:
        doc = naive_summarize(doc, max_chars=max(4000, SAFE_PROMPT_CHAR_LIMIT // 3))
        remaining = SAFE_PROMPT_CHAR_LIMIT - len(doc)

    if remaining > 0 and refs:
        per_ref_budget = max(1200, remaining // max(1, len(refs)))
        for r in refs:
            # Skip the "No References" placeholder when summarizing
            if r.get("name") != "No References":
                summarized_refs.append({
                    "name": r.get("name", "reference.txt"),
                    "content": naive_summarize(r.get("content", ""), per_ref_budget),
                })
    else:
        summarized_refs = refs

    return doc, summarized_refs


# -------------------------
# Prompt builder
# -------------------------
def build_prompt_for_canvas_editing(instruction: str, selected_text: str, document_text: str, references: List[Dict[str, str]]) -> str:
    """Build a specialized prompt for canvas-style editing of selected text."""
    document_text_final, references_final = gather_context(document_text, references)
    
    lines = [
        "You are an AI text editor. Your task is to edit the selected text according to the user's instruction.",
        "IMPORTANT: Provide ONLY the edited text as your response. Do not include explanations, quotes, or any other content.",
        "",
        f"Editing instruction: {instruction}",
        "",
        "Text to edit:",
        selected_text,
        "",
        "Document context (for reference):",
        document_text_final[:1000] + "..." if len(document_text_final) > 1000 else document_text_final,
        ""
    ]
    
    if references_final:
        lines.append("Reference materials:")
        for idx, r in enumerate(references_final[:2], start=1):  # Limit to 2 refs for context
            name = r.get("name", f"reference_{idx}.txt")
            content = r.get("content", "")[:500] + "..." if len(r.get("content", "")) > 500 else r.get("content", "")
            lines.append(f"--- {name} ---")
            lines.append(content)
        lines.append("")
    
    lines.append("Remember: Provide ONLY the edited text that should replace the selected text.")
    
    return "\n".join(lines)


def build_prompt(user_message: str, document_text: str, references: List[Dict[str, str]], selected_text: Optional[str], enable_document_editing: bool = False) -> str:
    system_preamble = (
        "You are an AI document assistant inside a three-pane workspace. "
        "Use the user's document and any uploaded text references to answer. "
        "If a 'User selection' is provided, prioritize it when relevant. "
        "Do not overfit to truncated previews; reason holistically and admit uncertainty if needed."
    )
    
    if enable_document_editing:
        system_preamble += (
            "\n\nYou can modify the document content when requested. "
            "When updating document content, use the format: "
            "```DOCUMENT_UPDATE\n[new content]\n``` "
            "This will replace the entire document content. "
            "For partial updates, include the context around changes."
        )

    document_text_final, references_final = gather_context(document_text, references)

    lines: List[str] = []
    lines.append(f"SYSTEM:\n{system_preamble}\n")

    if selected_text:
        lines.append("User selection (quoted):\n\n" + "\n".join([f"> {l}" for l in selected_text.splitlines()]))
        lines.append("")

    lines.append("Current Document:\n" + document_text_final)
    lines.append("")

    if references_final:
        # Check if we have Qdrant references
        qdrant_refs = [r for r in references_final if r.get("name", "").startswith("Qdrant:")]
        manual_refs = [r for r in references_final if not r.get("name", "").startswith("Qdrant:") and r.get("name") != "No References"]
        
        if qdrant_refs:
            lines.append("AI-Searchable References (Qdrant):")
            for idx, r in enumerate(qdrant_refs, start=1):
                name = r.get("name", f"qdrant_reference_{idx}")
                content = r.get("content", "")
                lines.append(f"--- Reference {idx}: {name} ---\n{content}")
            lines.append("")
        
        if manual_refs:
            lines.append("Uploaded References:")
            for idx, r in enumerate(manual_refs, start=1):
                name = r.get("name", f"reference_{idx}.txt")
                content = r.get("content", "")
                lines.append(f"--- Reference {idx}: {name} ---\n{content}")
            lines.append("")

    lines.append("User message:\n" + (user_message or ""))
    return "\n\n".join(lines)


# -------------------------
# Gemini client
# -------------------------
def configure_gemini() -> None:
    if not HAS_GEMINI:
        return
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)


def call_gemini_stream(prompt: str) -> Generator[str, None, None]:
    if not HAS_GEMINI:
        yield "[Gemini SDK not installed. Please install google-generativeai]"
        return
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt], stream=True)
        for event in response:
            text = getattr(event, "text", None)
            if text:
                yield text
    except Exception as exc:
        yield f"[Error from Gemini: {exc}]"


def extract_document_update(text: str) -> Optional[str]:
    """Extract document update from AI response if present."""
    import re
    pattern = r'```DOCUMENT_UPDATE\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def process_ai_response(response_text: str) -> Tuple[str, Optional[str]]:
    """Process AI response and extract any document updates."""
    document_update = extract_document_update(response_text)
    if document_update:
        # Remove the document update block from the displayed response
        import re
        cleaned_response = re.sub(r'```DOCUMENT_UPDATE\n.*?\n```\s*', '', response_text, flags=re.DOTALL)
        cleaned_response = cleaned_response.strip()
        if not cleaned_response:
            cleaned_response = "✅ Document updated successfully."
        return cleaned_response, document_update
    return response_text, None


# -------------------------
# UI helpers
# -------------------------
def sticky_header(title: str, pane_key: str, right_content: str = "") -> None:
    is_collapsed = st.session_state.collapsed[pane_key]
    
    # Create header with title and buttons
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.markdown(f"<div class='pane-title'>{title}</div>", unsafe_allow_html=True)
    
    with header_col2:
        # Right content (like refresh button)
        if right_content:
            st.markdown(right_content, unsafe_allow_html=True)
        
        # Collapse button
        collapse_btn_text = "🔽" if is_collapsed else "🔼"
        if st.button(collapse_btn_text, key=f"collapse_{pane_key}", help=f"Toggle {pane_key} pane"):
            st.session_state.collapsed[pane_key] = not st.session_state.collapsed[pane_key]
            st.rerun()


def focus_chat_input_js() -> None:
    # Try multiple selectors to focus chat input
    st.components.v1.html(
        """
        <script>
          const tryFocus = () => {
            const candidates = [
              'textarea[aria-label="chat input"]',
              'textarea[aria-label="Chat message"]',
              'div[data-baseweb="textarea"] textarea',
              'textarea'
            ];
            for (const sel of candidates) {
              const el = document.querySelector(sel);
              if (el) { el.focus(); return; }
            }
          };
          setTimeout(tryFocus, 50);
          setTimeout(tryFocus, 250);
          setTimeout(tryFocus, 600);
        </script>
        """,
        height=0,
    )


# Removed sidebar controls - keeping only the three panes


# -------------------------
# Pane renderers
# -------------------------
def render_chat_pane() -> None:
    col = st.container()
    chat_is_collapsed = st.session_state.collapsed["chat"]
    
    if chat_is_collapsed:
        sticky_header("💬 Chat", "chat")
        return
    
    sticky_header("💬 Chat", "chat")

    # Chat mode toggle buttons
    st.markdown("**Chat Mode:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Discuss", 
                    type="primary" if st.session_state.chat_mode == "discuss" else "secondary",
                    use_container_width=True,
                    help="Chat normally using document and references as context"):
            st.session_state.chat_mode = "discuss"
            st.rerun()
    with col2:
        if st.button("✏️ Update", 
                    type="primary" if st.session_state.chat_mode == "update" else "secondary",
                    use_container_width=True,
                    help="AI will update the document based on your instructions"):
            st.session_state.chat_mode = "update"
            st.rerun()
    
    # Show current mode
    mode_emoji = "💬" if st.session_state.chat_mode == "discuss" else "✏️"
    mode_name = "Discuss" if st.session_state.chat_mode == "discuss" else "Update"
    st.info(f"{mode_emoji} **{mode_name} Mode Active** - {('AI will chat using document/references as context' if st.session_state.chat_mode == 'discuss' else 'AI will update the document based on your instructions')}")
    
    # Show current Qdrant references being used as context
    qdrant_refs = st.session_state.get("qdrant_references", [])
    if qdrant_refs:
        with st.expander(f"🔍 **AI-Searchable Context** ({len(qdrant_refs)} documents)", expanded=False):
            st.markdown("**These documents are currently being used as context for your chat:**")
            for idx, ref in enumerate(qdrant_refs):
                doc_id = ref.get("doc_id", f"doc_{idx}")
                score = ref.get("score", 0)
                content_preview = ref.get("content", "")[:150] + "..." if len(ref.get("content", "")) > 150 else ref.get("content", "")
                st.markdown(f"**📄 {doc_id}** (Relevance: {score:.3f})")
                st.text(content_preview)
                st.divider()
    elif st.session_state.get("qdrant_initialized", False):
        st.info("💡 **No AI-searchable documents found.** Use the 'Find Resources' button in the References pane to search for relevant documents.")

    # Render chat history
    for idx, turn in enumerate(st.session_state.chat_history):
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(turn.get("content", ""))

    # Clear chat button
    clear = st.button("Clear chat", use_container_width=True)
    if clear:
        st.session_state.chat_history = []
        st.rerun()

    # Chat input
    qdrant_refs = st.session_state.get("qdrant_references", [])
    if qdrant_refs:
        placeholder_text = ("Enter instructions to update the document using the found references..." if st.session_state.chat_mode == "update" 
                           else f"Ask about the document or the {len(qdrant_refs)} found references...")
    else:
        placeholder_text = ("Enter instructions to update the document..." if st.session_state.chat_mode == "update" 
                           else "Ask about the document or references...")
    
    user_message = st.chat_input(placeholder_text, key="chat_input")
    
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Build prompt based on mode
        if st.session_state.chat_mode == "update":
            # Update mode: AI should always update the document
            prompt = build_prompt(
                user_message=user_message,
                document_text=st.session_state.document_text,
                references=st.session_state.references,
                selected_text=st.session_state.selected_text,
                enable_document_editing=True,
            )
        else:
            # Discuss mode: AI should just chat
            prompt = build_prompt(
                user_message=user_message,
                document_text=st.session_state.document_text,
                references=st.session_state.references,
                selected_text=st.session_state.selected_text,
                enable_document_editing=False,
            )

        # Stream model response
        with st.chat_message("assistant"):
            stream = call_gemini_stream(prompt)
            full_text = st.write_stream(stream)
        
        # Handle response based on mode
        if st.session_state.chat_mode == "update":
            # In update mode, try to extract document update or use the response as new content
            cleaned_response, document_update = process_ai_response(full_text or "")
            
            if document_update:
                # Formal DOCUMENT_UPDATE format
                old_length = len(st.session_state.document_text)
                st.session_state.document_text = document_update
                new_length = len(st.session_state.document_text)
                st.success(f"📝 Document updated! ({old_length} → {new_length} chars)")
                response_to_save = f"✅ Updated the document as requested."
            else:
                # Use the AI response as the new document content
                old_length = len(st.session_state.document_text)
                st.session_state.document_text = full_text or ""
                new_length = len(st.session_state.document_text)
                st.success(f"📝 Document replaced with AI response! ({old_length} → {new_length} chars)")
                response_to_save = f"✅ Replaced document content with new text."
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_to_save})
        else:
            # In discuss mode, just save the response normally
            st.session_state.chat_history.append({"role": "assistant", "content": full_text or ""})
        
        st.rerun()


def render_document_pane() -> None:
    doc_is_collapsed = st.session_state.collapsed["doc"]
    if doc_is_collapsed:
        sticky_header("📝 Document", "doc")
        return

    sticky_header("📝 Document", "doc")

    selected_text: Optional[str] = None
    updated_text: Optional[str] = None

    # Document header with save button
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("**Document Content**")
    with col2:
        if st.button("💾 Save", key="save_doc_btn", help="Save document", use_container_width=True):
            st.success("Document saved!")
            st.rerun()
    with col3:
        if HAS_QDRANT and st.button("🗄️ Add to Qdrant", key="add_to_qdrant_btn", help="Add document to vector database", use_container_width=True):
            if st.session_state.document_text.strip():
                # Generate a numeric ID for Qdrant (it requires integers or UUIDs)
                doc_id = abs(hash(st.session_state.document_text)) % 1000000  # Use hash modulo for unique numeric ID
                if add_document_to_qdrant(doc_id, st.session_state.document_text):
                    st.success("✅ Document added to Qdrant!")
                else:
                    st.error("❌ Failed to add document to Qdrant")
            else:
                st.warning("Please add some text to the document first")
    
    # Simple manual text selection for chat context
    st.markdown("**Manual Text Selection for Chat:**")
    manual_selection = st.text_input(
        "Copy and paste text here to use as context in chat:",
        value=st.session_state.get("selected_text", ""),
        key="manual_selection",
        placeholder="Paste text here to use as context for the chatbot...",
        help="Copy text from the document below and paste it here to use as context in your chat"
    )
    
    # Update session state with manual selection
    if manual_selection and manual_selection.strip():
        st.session_state.selected_text = manual_selection.strip()
    else:
        st.session_state.selected_text = None

    # Main document text area
    updated_text = st.text_area(
        "Type or paste your document content here",
        value=st.session_state.document_text,
        height=400,
        key="document_textarea",
        help="This is your main document. Copy text to the selection box above to use as chat context.",
        label_visibility="collapsed"
    )



    # Persist document text and detect changes
    if updated_text is not None and updated_text != st.session_state.document_text:
        st.session_state.document_text = updated_text
    
    # Show current selection info if any
    if st.session_state.selected_text:
        with st.container(border=True):
            st.markdown("**✅ Selected Text for Chat Context:**")
            preview = st.session_state.selected_text[:200] + "..." if len(st.session_state.selected_text) > 200 else st.session_state.selected_text
            st.markdown(f"> {preview}")
            st.caption("This text will be included as context in your chat messages.")
    
    # Show how to use the new system
    st.info("💡 **How to use:** \n- **Discuss Mode**: Chat normally about the document \n- **Update Mode**: AI will update the document based on your instructions \n- Copy text to the selection box above to use as chat context")

    words, chars = count_words_and_chars(st.session_state.document_text)
    st.caption(f"Words: {words} • Characters: {chars}")


def render_reference_pane() -> None:
    ref_is_collapsed = st.session_state.collapsed["ref"]
    if ref_is_collapsed:
        sticky_header("📁 References", "ref")
        return

    sticky_header("📁 References", "ref")

    # Create tabs for different reference types
    tab1, tab2 = st.tabs(["📁 Uploads", "🔍 Qdrant References"])
    
    # Tab 1: Manual Uploads (existing functionality)
    with tab1:
        # Hide the default file uploader since we have our custom button
        uploaded_files = None
        if "file_uploader_trigger" in st.session_state and st.session_state.file_uploader_trigger:
            uploaded_files = st.file_uploader(
                "Upload .txt files",
                accept_multiple_files=True,
                type=["txt"],
                key="upload_files_hidden",
                label_visibility="hidden"
            )
            if uploaded_files:
                st.session_state.file_uploader_trigger = False
                refs: List[Dict[str, str]] = []
                for f in uploaded_files:
                    try:
                        content = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        content = ""
                    refs.append({"name": f.name, "content": content})
                st.session_state.references = refs

        # Render previews
        if not st.session_state.references:
            st.markdown('<div class="empty-references"><div class="upload-prompt">', unsafe_allow_html=True)
            if st.button("📁 Upload Document", key="custom_upload_btn", use_container_width=True, type="primary"):
                st.session_state.file_uploader_trigger = True
                st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)
        else:
            for idx, r in enumerate(st.session_state.references):
                name = r.get("name", f"reference_{idx}.txt")
                content = r.get("content", "")
                preview, truncated = truncate_for_preview(content, TRUNCATED_PREVIEW_CHARS)
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.text(preview)
                    cols = st.columns([1, 1, 6])
                    with cols[0]:
                        if st.button("Copy preview", key=f"copy_prev_{idx}"):
                            # Best-effort copy via a tiny JS snippet
                            st.components.v1.html(
                                f"""
                                <script>
                                  navigator.clipboard.writeText({preview!r});
                                </script>
                                """,
                                height=0,
                            )
                            st.toast("Preview copied to clipboard")
                    with cols[1]:
                        st.caption("Truncated" if truncated else "Full")
    
    # Tab 2: Qdrant References (new functionality)
    with tab2:
        if not HAS_QDRANT:
            st.error("Qdrant is not available. Please install dependencies.")
            return
            
        # Initialize Qdrant if not done yet
        if not st.session_state.qdrant_initialized:
            with st.spinner("Initializing Qdrant..."):
                if qdrant_manager.initialize():
                    st.session_state.qdrant_initialized = True
                    st.success("✅ Qdrant connected successfully!")
                else:
                    st.error("❌ Failed to connect to Qdrant. Please check your setup.")
                    return
        
        # Show Qdrant status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**🔍 Find Relevant Resources**")
            st.markdown("Search for documents similar to your current text using AI-powered semantic search.")
        
        with col2:
             if st.button("🔍 Find Resources", key="find_resources_btn", use_container_width=True, type="primary"):
                 if st.session_state.document_text.strip():
                     with st.spinner("Searching for relevant documents..."):
                         # Search Qdrant for similar documents
                         query = st.session_state.document_text[:500]  # Use first 500 chars as query
                         results = search_qdrant(query, top_k=5)
                         st.session_state.qdrant_references = results
                         
                         # Show search debug info
                         if results:
                             st.success(f"Found {len(results)} relevant documents!")
                             
                             # Debug: Show what was searched
                             with st.expander("🔍 Search Debug Info", expanded=False):
                                 st.write(f"**Query:** {query[:100]}{'...' if len(query) > 100 else ''}")
                                 st.write(f"**Query length:** {len(query)} characters")
                                 st.write(f"**Results found:** {len(results)}")
                                 
                                 # Show top result details
                                 if results:
                                     top_result = results[0]
                                     st.write(f"**Top result score:** {top_result.get('score', 0):.3f}")
                                     st.write(f"**Top result ID:** {top_result.get('doc_id', 'N/A')}")
                         else:
                             st.info("No relevant documents found. Try adding some documents to Qdrant first.")
                             
                             # Debug: Show why no results
                             with st.expander("🔍 Search Debug Info", expanded=False):
                                 st.write(f"**Query:** {query[:100]}{'...' if len(query) > 100 else ''}")
                                 st.write(f"**Query length:** {len(query)} characters")
                                 st.write("**Possible reasons:**")
                                 st.write("- Query too short or generic")
                                 st.write("- No semantic similarity to existing documents")
                                 st.write("- Score threshold too high")
                 else:
                     st.warning("Please add some text to the document pane first.")
        
        # Display Qdrant search results
        if st.session_state.qdrant_references:
            st.markdown("**📚 Found Documents:**")
            for idx, result in enumerate(st.session_state.qdrant_references):
                with st.container(border=True):
                    doc_id = result.get("doc_id", f"doc_{idx}")
                    content = result.get("content", "")
                    score = result.get("score", 0)
                    metadata = result.get("metadata", {})
                    
                    # Header with score
                    st.markdown(f"**📄 {doc_id}** (Similarity: {score:.3f})")
                    
                    # Content preview
                    preview, truncated = truncate_for_preview(content, TRUNCATED_PREVIEW_CHARS)
                    st.text(preview)
                    
                    # Metadata info
                    if metadata:
                        with st.expander("📊 Metadata"):
                            for key, value in metadata.items():
                                st.write(f"**{key}:** {value}")
                    
                    # Actions
                    cols = st.columns([1, 1, 1, 3])
                    with cols[0]:
                        if st.button("Copy", key=f"copy_qdrant_{idx}"):
                            st.components.v1.html(
                                f"""
                                <script>
                                  navigator.clipboard.writeText({content!r});
                                </script>
                                """,
                                height=0,
                            )
                            st.toast("Content copied to clipboard")
                    
                    with cols[1]:
                        if st.button("Use as Context", key=f"use_context_{idx}"):
                            st.session_state.selected_text = content
                            st.success("Added to chat context!")
                    
                    with cols[2]:
                        st.caption("Truncated" if truncated else "Full")
        else:
            st.info("💡 Click 'Find Resources' to search for relevant documents based on your current text.")
            
            # Show collection info
            if st.session_state.qdrant_initialized:
                with st.expander("📊 Qdrant Collection Info"):
                    info = qdrant_manager.get_collection_info()
                    if info:
                        st.write(f"**Collection:** {info.get('name', 'N/A')}")
                        st.write(f"**Vector Size:** {info.get('vector_size', 'N/A')}")
                        st.write(f"**Documents:** {info.get('points_count', 'N/A')}")
                    else:
                        st.write("Collection info not available")
            
            # Show usage instructions
            st.info("""
            **💡 How to use AI-Searchable References:**
            
            1. **Search**: Type text in the Document pane and click "Find Resources"
            2. **Chat**: Found documents automatically become context for the chatbot
            3. **Discuss Mode**: Ask questions about the found documents
            4. **Update Mode**: Use the documents as reference to update your text
            
            **Example queries to try:**
            - "What are the main points about AI in the found documents?"
            - "Summarize the programming concepts mentioned"
            - "How do the documents relate to climate change?"
            """)


# -------------------------
# Main app
# -------------------------
def main() -> None:
    # Enable wide mode by default
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_dotenv(override=False)
    ensure_state_defaults()
    load_styles()
    configure_gemini()

    # Global header spacing
    st.markdown("<div class='app-spacer'></div>", unsafe_allow_html=True)

    # Columns with dynamic ratios based on collapsed state
    chat_collapsed = st.session_state.collapsed["chat"]
    doc_collapsed = st.session_state.collapsed["doc"]
    ref_collapsed = st.session_state.collapsed["ref"]
    
    # Calculate ratios - collapsed panes get minimal space
    base_ratios = [
        0.05 if chat_collapsed else st.session_state.pane_widths["chat"],
        0.05 if doc_collapsed else st.session_state.pane_widths["doc"],
        0.05 if ref_collapsed else st.session_state.pane_widths["ref"],
    ]
    
    # Normalize ratios for non-collapsed panes
    total_collapsed = sum([0.05 if collapsed else 0 for collapsed in [chat_collapsed, doc_collapsed, ref_collapsed]])
    total_expanded = sum([ratio for ratio, collapsed in zip([st.session_state.pane_widths["chat"], st.session_state.pane_widths["doc"], st.session_state.pane_widths["ref"]], [chat_collapsed, doc_collapsed, ref_collapsed]) if not collapsed])
    
    if total_expanded > 0:
        expansion_factor = (1.0 - total_collapsed) / total_expanded
        ratios = [
            0.05 if chat_collapsed else st.session_state.pane_widths["chat"] * expansion_factor,
            0.05 if doc_collapsed else st.session_state.pane_widths["doc"] * expansion_factor,
            0.05 if ref_collapsed else st.session_state.pane_widths["ref"] * expansion_factor,
        ]
    else:
        ratios = base_ratios
    
    col_chat, col_doc, col_ref = st.columns(ratios, gap="small")

    with col_chat:
        render_chat_pane()
    with col_doc:
        render_document_pane()
    with col_ref:
        render_reference_pane()

    # API key warning
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY not found in environment. Create a .env file to enable model responses.")
    
    # Debug info (can be removed later)
    with st.expander("🔍 Debug Info", expanded=False):
        st.write("**Document length:**", len(st.session_state.document_text))
        st.write("**References count:**", len(st.session_state.references))
        if st.session_state.references:
            for i, ref in enumerate(st.session_state.references):
                st.write(f"**Reference {i+1}:** {ref.get('name', 'Unknown')} ({len(ref.get('content', ''))} chars)")
        if st.session_state.selected_text:
            st.write("**Selected text:**", f"'{st.session_state.selected_text[:100]}...'" if len(st.session_state.selected_text) > 100 else f"'{st.session_state.selected_text}'")


if __name__ == "__main__":
    main()


