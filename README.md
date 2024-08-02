# RAG Indexing Pipeline in Rust

This repository contains the source code for my [post](https://medium.com/towards-data-science/scale-up-your-rag-a-rust-powered-indexing-pipeline-with-lancedb-and-candle-cc681c6162e8): *Scale Up Your RAG: A Rust-Powered Indexing Pipeline with LanceDB and Candle*  that was published in **Towards Data Science**. The post explains how to build a high-performance Retrieval-Augmented Generation (RAG) indexing pipeline implemented in Rust.
It also demonstrates how to efficiently read, chunk, embed, and store textual documents as vectors using HuggingFace's Candle framework and LanceDB.

## Features

- Fast document processing and chunking
- Efficient text embedding using Candle
- Vector storage with LanceDB
- Scalable design for handling large volumes of data
- Standalone application deployable in various environments

## Prerequisites

- Rust (latest stable version)
- Cargo (Rust's package manager)

## Usage

Clone this repository:
```
git clone https://github.com/your-username/rag-indexing-pipeline-rust.git
cd rag-indexing-pipeline-rust
```
To run the app on test data, use the following command:
```
cargo run --release -- --input-directory embedding_files_test --db-uri data/vecdb1
```
