pub mod models;
pub mod storage;
use candle_nn::embedding;
use clap::Parser;
use models::bert::BertModelWrapper;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;
use tracing::{info, warn};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[clap(short, long)]
    input_directory: String,
    #[clap(short, long)]
    db_uri: String,
}

struct EmbeddingEntry {
    filename: String,
    embedding: Vec<f32>,
}

// constants for the model id and default revision
const EMB_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const EMB_MODEL_REV: &str = "refs/pr/21";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli_args = Cli::parse();
    let ts_mark = std::time::Instant::now();

    // init the channel that sends data to the thread that writes embedding to the db
    let (sender, reciever) = std::sync::mpsc::channel::<EmbeddingEntry>();
    let db_writer_task =
        init_db_writer_task(reciever, cli_args.db_uri.as_str(), "vectors_table_1", 100).await?;

    // list the files in the directory to be embedded
    let files_dir = fs::read_dir(cli_args.input_directory)?;
    files_dir
        .into_iter()
        .map(|file| file.unwrap().path().to_str().unwrap().to_string())
        .collect::<Vec<String>>()
        .into_par_iter()
        .for_each(|filename| {
            if let Err(e) = process_text_file(sender.clone(), filename.as_str()) {
                warn!("Error processing file: {}: Error:{}", filename, e)
            }
        });

    drop(sender); // this will close the channel
    db_writer_task.await?; // wait for the db writer task to finish before exiting
    info!("Total time taken: {:?}", ts_mark.elapsed());
    Ok(())
}

fn process_text_file(sender: Sender<EmbeddingEntry>, filename: &str) -> anyhow::Result<()> {
    let bert_model = models::bert::get_model_reference(EMB_MODEL_ID, EMB_MODEL_REV)?;
    info!("reading file: {}", filename);
    let text_chunks = read_file_in_chunks(filename, 256)?;
    let text_chunks: Vec<&str> = text_chunks.iter().map(AsRef::as_ref).collect();
    let file_vector = embed_multiple_sentences(&text_chunks, false, &bert_model)?;
    sender.send(EmbeddingEntry {
        filename: filename.to_string(),
        embedding: file_vector[0].clone(),
    })?;
    Ok(())
}

/// Initialize the task that writes the embeddings to the db
/// ## Arguments
/// * reciever: the channel that receives the embeddings
/// * db_uri: the uri of the db e.g. data/vecdb
/// * table_name: the name of the table to write the embeddings to
async fn init_db_writer_task(
    reciever: Receiver<EmbeddingEntry>,
    db_uri: &str,
    table_name: &str,
    buffer_size: usize,
) -> anyhow::Result<JoinHandle<()>> {
    let db = storage::VecDB::connect(db_uri, table_name).await?;
    let task_handle = tokio::spawn(async move {
        let mut embeddings_buffer = Vec::new();
        while let Ok(embedding) = reciever.recv() {
            embeddings_buffer.push(embedding);
            if embeddings_buffer.len() >= buffer_size {
                let (keys, vectors) = extract_keys_and_vectors(&embeddings_buffer);
                db.add_vector(&keys, vectors, 384).await.unwrap();
                embeddings_buffer.clear();
            }
        }
        if !embeddings_buffer.is_empty() {
            let (keys, vectors) = extract_keys_and_vectors(&embeddings_buffer);
            db.add_vector(&keys, vectors, 384).await.unwrap();
        }
    });
    Ok(task_handle)
}

fn extract_keys_and_vectors(embeddings_buffer: &[EmbeddingEntry]) -> (Vec<&str>, Vec<Vec<f32>>) {
    embeddings_buffer
        .iter()
        .map(|entry| (entry.filename.as_str(), entry.embedding.clone()))
        .unzip::<&str, Vec<f32>, Vec<&str>, Vec<Vec<f32>>>()
}

fn embed_multiple_sentences(
    sentences: &[&str],
    apply_mean: bool,
    bert_model: &BertModelWrapper,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let multiple_embeddings = bert_model.embed_sentences(sentences, apply_mean)?;
    if apply_mean {
        let multiple_embeddings = multiple_embeddings.to_vec1::<f32>()?;
        Ok(vec![multiple_embeddings])
    } else {
        let multiple_embeddings = multiple_embeddings.to_vec2::<f32>()?;
        Ok(multiple_embeddings)
    }
}

fn embed_sentence(sentence: &str, bert_model: &BertModelWrapper) -> anyhow::Result<Vec<f32>> {
    let embedding = bert_model.embed_sentence(sentence)?;
    println!("embedding Tensor: {:?}", embedding);
    // we squeeze the tensor to remove the batch dimension
    let embedding = embedding.squeeze(0)?;
    println!("embedding Tensor after squeeze: {:?}", embedding);
    let embedding = embedding.to_vec1::<f32>().unwrap();
    //println!("embedding Vec: {:?}", embedding);
    Ok(embedding)
}

fn init_tracing() {
    if let Ok(level_filter) = tracing_subscriber::EnvFilter::try_from_env("LOG_LEVEL") {
        tracing_subscriber::fmt()
            .with_env_filter(level_filter)
            .with_ansi(true)
            .with_file(true)
            .with_line_number(true)
            .init();
    } else {
        println!("Failed to parse LOG_LEVEL env variable, using default log level: INFO");
        tracing_subscriber::fmt()
            .with_ansi(true)
            .with_file(true)
            .with_line_number(true)
            .init();
    }
}

fn read_file_in_chunks(file_path: &str, chunk_size: usize) -> anyhow::Result<Vec<String>> {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let mut sentences = Vec::new();
    let mut text_buffer = String::new();
    for text in reader.lines() {
        let text = text?;
        text_buffer.push_str(text.as_str());
        let word_count = text_buffer.split_whitespace().count();
        if word_count >= chunk_size {
            sentences.push(text_buffer.clone());
            text_buffer.clear();
        }
    }
    if !text_buffer.is_empty() {
        sentences.push(text_buffer.clone());
    }
    Ok(sentences)
}