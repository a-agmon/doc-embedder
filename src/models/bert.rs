use std::{cell::RefCell, rc::Rc};

use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::{debug, info};

// constants for the model id and default revision
const EMB_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const EMB_MODEL_REV: &str = "refs/pr/21";

thread_local! {
    static BERT_MODEL: Rc<BertModelWrapper> =   {
        info!("Loading a model on thread: {:?}", std::thread::current().id());
        let model = BertModelWrapper::new(candle::Device::Cpu, EMB_MODEL_ID, EMB_MODEL_REV);
        match model {
            Ok(model) => Rc::new(model),
            Err(e) => {
                panic!("Failed to load the model: {}", e);
            }
        }
    }
}

pub fn get_model_reference() -> anyhow::Result<Rc<BertModelWrapper>> {
    BERT_MODEL.with(|model| Ok(model.clone()))
}

pub struct BertModelWrapper {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl BertModelWrapper {
    pub fn new(device: Device, model_id: &str, revision: &str) -> anyhow::Result<Self> {
        let repo = Repo::with_revision(model_id.into(), RepoType::Model, revision.into());
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("model.safetensors")?;
        // load the model config
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        // load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
        // load the model
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed_sentence(&self, sentence: &str) -> anyhow::Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode(sentence, true)
            .map_err(anyhow::Error::msg)?;
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let start = std::time::Instant::now();
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        debug!("time taken for forward: {:?}", start.elapsed());
        debug!("embeddings: {:?}", embeddings);
        let embeddings = Self::apply_max_pooling(&embeddings)?;
        debug!("embeddings after pooling: {:?}", embeddings);
        let embeddings = Self::l2_normalize(&embeddings)?;
        Ok(embeddings)
    }

    pub fn embed_sentences(&self, sentences: &[&str], apply_mean: bool) -> anyhow::Result<Tensor> {
        let mut all_tokens = Vec::with_capacity(sentences.len());
        for sentence in sentences {
            let tokens = self
                .tokenizer
                .encode(*sentence, true)
                .map_err(anyhow::Error::msg)?;
            all_tokens.push(tokens);
        }

        let batch_size = sentences.len();
        let max_length = all_tokens[0].get_ids().len(); // Assuming all are padded to the same length

        let mut token_ids = Vec::with_capacity(batch_size * max_length);
        let mut attention_mask = Vec::with_capacity(batch_size * max_length);

        for tokens in all_tokens {
            token_ids.extend_from_slice(tokens.get_ids());
            attention_mask.extend_from_slice(tokens.get_attention_mask());
        }

        let token_ids = Tensor::new(token_ids, &self.device)?.reshape((batch_size, max_length))?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        let embeddings = Self::apply_mean_pooling(&embeddings)?;
        let embeddings = Self::l2_normalize(&embeddings)?;
        if apply_mean {
            let embeddings = Self::apply_mean_pooling(&embeddings)?;
            Ok(embeddings)
        } else {
            Ok(embeddings)
        }
    }

    pub fn apply_max_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        Ok(embeddings.max(1)?)
    }

    /// Apply mean pooling to the embeddings
    /// The input tensor should either have the shape (n_sentences, n_tokens, hidden_size) or (n_tokens, hidden_size)
    /// depending on whether the input is a batch of sentences or a single sentence
    pub fn apply_mean_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        match embeddings.rank() {
            3 => {
                let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
                (embeddings.sum(1)? / (n_tokens as f64)).map_err(anyhow::Error::msg)
            }
            2 => {
                let (n_tokens, _hidden_size) = embeddings.dims2()?;
                (embeddings.sum(0)? / (n_tokens as f64)).map_err(anyhow::Error::msg)
            }
            _ => anyhow::bail!("Unsupported tensor rank for mean pooling"),
        }
    }

    pub fn l2_normalize(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        let normalized = embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        //info!("normalized embeddings: {:?}", normalized);
        Ok(normalized)
    }

    fn cosine_similarity(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
        let sum_ij = (a * b)?.sum_all()?.to_scalar::<f64>()?;
        let sum_i2 = (a * a)?.sum_all()?.to_scalar::<f64>()?;
        let sum_j2 = (b * b)?.sum_all()?.to_scalar::<f64>()?;
        let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
        Ok(cosine_similarity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed_multiple_sentences;
    fn cosine_similarity_for_test(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        if a.len() != b.len() {
            panic!("Vectors must have the same length");
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

        dot_product / (magnitude_a * magnitude_b)
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Tensor::from_slice(&[1.0, 1.0, 2.0], (1, 3), &Device::Cpu).unwrap();
        let b = Tensor::from_slice(&[1.0, 1.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let c = Tensor::from_slice(&[2.0, 4.0, 5.0], (1, 3), &Device::Cpu).unwrap();
        // we will test that a and b are closer than a and c or b and c
        let similarity_ab = BertModelWrapper::cosine_similarity(&a, &b).unwrap();
        let similarity_ac = BertModelWrapper::cosine_similarity(&a, &c).unwrap();
        let similarity_bc = BertModelWrapper::cosine_similarity(&b, &c).unwrap();
        // just print all results now
        println!("similarity_ab: {}", similarity_ab);
        println!("similarity_ac: {}", similarity_ac);
        println!("similarity_bc: {}", similarity_bc);

        assert!(similarity_ab > similarity_ac);
        assert!(similarity_ab > similarity_bc);
    }

    #[test]
    fn test_embed_multiple_sentences() {
        let bert_model = get_model_reference().unwrap();
        let sentences_1 = vec![
            "My cat loves to nap in the sunlight",
            "Whiskers is always chasing after the red laser dot",
            "As a Siamese, my cat has a beautiful coat",
        ];

        let sentences_2 = vec![
            "My kitten enjoys playing with string",
            "The little feline is quick to pounce on toys",
            "Siamese kittens have striking blue eyes",
        ];

        let sentences_3 = vec![
            "The chef is preparing a gourmet meal tonight",
            "Cooking requires precise timing and skill",
            "Gourmet dishes often include exotic ingredients",
        ];
        // now we emebed and test the similarity
        let embeddings_1 = embed_multiple_sentences(&sentences_1, true, &bert_model).unwrap();
        let embeddings_2 = embed_multiple_sentences(&sentences_2, true, &bert_model).unwrap();
        let embeddings_3 = embed_multiple_sentences(&sentences_3, true, &bert_model).unwrap();
        // now we test the similarity between the embeddings
        let similarity_12 = cosine_similarity_for_test(&embeddings_1[0], &embeddings_2[0]);
        let similarity_13 = cosine_similarity_for_test(&embeddings_1[0], &embeddings_3[0]);
        let similarity_23 = cosine_similarity_for_test(&embeddings_2[0], &embeddings_3[0]);

        println!("similarity_12: {}", similarity_12);
        println!("similarity_13: {}", similarity_13);
        println!("similarity_23: {}", similarity_23);
        assert!(similarity_12 > similarity_13);
        assert!(similarity_12 > similarity_23);
    }
}
