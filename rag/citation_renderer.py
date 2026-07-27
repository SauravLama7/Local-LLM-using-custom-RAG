import re
import streamlit as st


def extract_citations(text: str) -> list[int]:
    """Extract all citation numbers [1], [2] etc from text."""
    return [int(n) for n in re.findall(r'\[(\d+)\]', text)]


def render_citations(full_response: str, indexed_chunks: dict, sources: list):
    """
    Render improved citations with popup chunk preview.
    Shows source badges and expandable chunk content.
    """
    if not sources and not indexed_chunks:
        return

    # Extract which citation numbers were actually used in the response
    used_indices = extract_citations(full_response)
    used_chunks  = {i: indexed_chunks[i] for i in used_indices if i in indexed_chunks}

    if not used_chunks and not sources:
        return

    st.markdown("---")
    st.markdown("**📚 Sources & Citations:**")

    # Group used chunks by source
    chunks_by_source = {}
    for idx, chunk_data in used_chunks.items():
        src = chunk_data["source"]
        if src not in chunks_by_source:
            chunks_by_source[src] = []
        chunks_by_source[src].append({"index": idx, "text": chunk_data["text"]})

    # Show sources not in used_chunks too
    for src in sources:
        if src not in chunks_by_source:
            chunks_by_source[src] = []

    # Render each source with expandable chunk preview
    for src, chunks in chunks_by_source.items():
        chunk_label = f"📄 {src}"
        if chunks:
            indices = [str(c["index"]) for c in chunks]
            chunk_label += f"  —  cited as [{', '.join(indices)}]"

        with st.expander(chunk_label):
            if chunks:
                for chunk in sorted(chunks, key=lambda x: x["index"]):
                    st.markdown(f"**[{chunk['index']}]**")
                    # Show chunk text in a styled box
                    st.markdown(
                        f"""<div style='
                            background-color: #1e1e2e;
                            border-left: 3px solid #7c3aed;
                            padding: 10px 14px;
                            border-radius: 4px;
                            font-size: 0.9em;
                            color: #cdd6f4;
                            margin-bottom: 8px;
                        '>{chunk["text"]}</div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("Source retrieved but no specific chunks cited inline.")