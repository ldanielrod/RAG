import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_DIR = Path("data/pdfs")
CHROMA_DIR = Path("storage/chroma")
COLLECTION = "policies_pdfs"

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_pdfs(pdf_dir: Path):
    if not pdf_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {pdf_dir.resolve()}")

    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No encontré PDFs en: {pdf_dir.resolve()}")

    docs = []
    print(f"📄 PDFs encontrados ({len(pdf_paths)}):")
    for p in pdf_paths:
        print(f" - {p.name}")

        loader = PyPDFLoader(str(p))
        pages = loader.load()  # Document por página

        # doc_id estable (hash del contenido) + metadata amigable
        doc_id = file_sha1(p)

        for d in pages:
            d.metadata["doc_id"] = doc_id
            d.metadata["source_file"] = p.name           # nombre original (puede ser feo, no importa)
            d.metadata["source_path"] = str(p.resolve()) # ruta completa
            # d.metadata["page"] ya viene del loader normalmente
        docs.extend(pages)

    return docs

def main():
    raw_docs = load_pdfs(PDF_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Reindex limpio (borra la colección si existe)
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    try:
        vectordb.delete_collection()
    except Exception:
        pass

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION
    )

    print(f"\n✅ Indexación lista.")
    print(f"   - Páginas: {len(raw_docs)}")
    print(f"   - Chunks:  {len(chunks)}")
    print(f"📦 DB: {CHROMA_DIR.resolve()}")
    print(f"🧾 Colección: {COLLECTION}")

if __name__ == "__main__":
    main()