import concurrent.futures
import os
import time
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import (CSVLoader, Docx2txtLoader, PyPDFLoader,
                                        UnstructuredFileLoader,
                                        UnstructuredHTMLLoader,
                                        UnstructuredMarkdownLoader,
                                        UnstructuredPowerPointLoader)
from langchain.text_splitter import TokenTextSplitter


# Chroma

class Ingest:
    def __init__(self, index_name, client, collection):
        self.index_name = index_name
        self.client = client
        self.collection = collection

    def load_file(self, file_path) -> List[Document]:
        _, file_extension = os.path.splitext(file_path)
        data: List[Document]
        if file_extension.lower() == '.html':
            loader = UnstructuredHTMLLoader(file_path)
            return loader.load()
        elif file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_extension.lower() == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()
        elif file_extension.lower() == '.csv':
            loader = CSVLoader(file_path)
            return loader.load()
        elif file_extension.lower() == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
            return loader.load()
        elif file_extension.lower() == '.docx':
            loader = Docx2txtLoader(file_path)
            return loader.load()
        else:
            # Perform action for other files or skip
            return UnstructuredFileLoader(file_path).load()

    def process_file(self, file_path, chunk_size=1000, chunk_overlap=400):
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        try:
            data = self.load_file(file_path=file_path)
            texts = text_splitter.split_documents(data)
            contents = [d.page_content for d in texts]
            metadatas = [d.metadata for d in texts]
            ids = [str(idx) for idx, d in enumerate(texts)]
            self.collection.add(documents=contents, metadatas=metadatas, ids=ids)
            # split and load into vector db
            print(f"Found {len(data)} parts in file {file_path}")
        except Exception as e:
            print(f"process_file: Error parsing file {file_path}.  {e}")

    def process_directory(self, folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self.process_file(file_path)
                except Exception as e:
                    print(f"process_directory: Error processing file {file_path}: {e}")

    def process_item(self, item):
        print(f"Processing item: {item}")
        self.process_file(item)
        # Add your processing logic here

    def process_items(self, items):
        print(f"Processing items: {items}")
        for i in items:
            self.process_file(i)

    def worker(self, queue):
        while True:
            item = queue.get()
            if item is None:
                break
            self.process_item(item)
            queue.task_done()

    def process(self, item_queue, total_items):
        max_threads = 24
        starttime = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(self.process_items, item_queue)
        print('That took {} seconds'.format(time.time() - starttime))
        print(f"processing a total of {total_items} of files")

    def load_data(self, folder_path):
        item_queue = []
        total_items = 0

        # Add items to the queue
        for root, _, files in os.walk(folder_path):
            for file in files:
                group = []
                file_path = os.path.join(root, file)
                _, file_extension = os.path.splitext(file_path)
                # only do file types we want to process
                if file_extension.lower() in ('.pdf', '.md', '.txt', '.html', '.pptx', '.docx'):
                    group.append(file_path)
                    total_items = total_items + 1
                item_queue.append(group)

        self.process(item_queue, total_items)