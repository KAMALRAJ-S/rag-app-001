import os
import shutil
import tempfile

import streamlit as st

from vectorstore_builder import build_vectorstore_from_uploads
from rag_chat import ask_question

st.set_page_config(page_title="RAG Assistant", page_icon="📄")
st.title("RAG Assistant")
st.write("Upload your own files, then ask questions about them.")

# Per-visitor session state — Streamlit already scopes st.session_state per browser session,
# so nobody else can see or query another visitor's uploads or chat history.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of (question, answer) for rendering

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB total, matches vectorstore_builder's limit

st.subheader("1. Upload files")

already_processed = st.session_state.vectorstore is not None

if already_processed:
    st.info(
        "Files for this session are already processed. To upload a different set of files, "
        "refresh the page to start a new session."
    )

uploaded_files = st.file_uploader(
    "Upload files (.txt, .pdf, .docx, .csv, .xlsx) — 100MB total limit",
    type=["txt", "pdf", "docx", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    disabled=already_processed,
)

if st.button("Process files", disabled=already_processed):
    if not uploaded_files:
        st.warning("Please choose at least one file first.")
    else:
        total_size = sum(f.size for f in uploaded_files)
        if total_size > MAX_UPLOAD_BYTES:
            st.error(
                f"Total upload size is {total_size / (1024*1024):.1f}MB, "
                f"which is over the 100MB limit. Please upload fewer or smaller files."
            )
        else:
            # Streamlit's uploader gives file-like objects in memory, not disk paths.
            # Our loaders expect real file paths, so write each to a temp file first.
            temp_paths = []
            with st.spinner("Processing files..."):
                tmp_dir = tempfile.mkdtemp()
                for f in uploaded_files:
                    temp_path = os.path.join(tmp_dir, f.name)
                    with open(temp_path, "wb") as out:
                        out.write(f.getbuffer())
                    temp_paths.append(temp_path)

                try:
                    vectorstore, skipped = build_vectorstore_from_uploads(temp_paths)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chat_history = []
                    st.session_state.chat_display = []

                    processed_count = len(temp_paths) - len(skipped)
                    st.success(f"Processed {processed_count} of {len(temp_paths)} file(s). You can ask questions now.")
                    if skipped:
                        skipped_list = "; ".join(f"{os.path.basename(p)} ({reason})" for p, reason in skipped)
                        st.warning(f"Skipped: {skipped_list}")
                except Exception as e:
                    st.error(f"Couldn't process those files: {e}")
                finally:
                    # The raw uploaded files are no longer needed once embedded into
                    # the vector store — clean up so temp files don't pile up on disk
                    # every time "Process files" is clicked.
                    shutil.rmtree(tmp_dir, ignore_errors=True)

st.subheader("2. Ask questions")

for question, answer in st.session_state.chat_display:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

user_question = st.chat_input("Type your question...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.write("Please upload and process at least one file before asking a question.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, updated_history = ask_question(
                    user_question, st.session_state.chat_history, st.session_state.vectorstore
                )
                st.session_state.chat_history = updated_history
                st.write(answer)

        st.session_state.chat_display.append((user_question, answer))
