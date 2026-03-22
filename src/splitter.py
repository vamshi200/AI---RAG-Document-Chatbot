from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)