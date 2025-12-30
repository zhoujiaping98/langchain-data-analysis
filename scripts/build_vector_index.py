from __future__ import annotations

from app.vector_index import rebuild_vector_index


def main():
    rebuild_vector_index()
    print("OK: rebuilt Chroma vector index in CHROMA_DIR.")


if __name__ == "__main__":
    main()
