import os
from src.rag_core import RAGEngine, DEBUG

rag = RAGEngine()

def main():
    print("🧠 RAG local listo. ENTER para salir.")
    while True:
        q = input("\nPregunta> ").strip()
        if not q:
            break
        out = rag.ask(q)
        print("\nRespuesta:\n", out["answer"])
        if out["citations"]:
            print("\nFuentes:")
            for c in out["citations"]:
                p = f"(pág. {c['page']})" if c.get("page") else ""
                print(f"- {c['file']} {p}".strip())
        else:
            print("\nFuentes:\n- (sin fuentes)")

if __name__ == "__main__":
    main()