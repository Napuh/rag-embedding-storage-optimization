<h1 align="center"> Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques</h1>

> Naamán Huerga-Pérez, Rubén Álvarez, Rubén Ferrero-Guillén, Alberto Martínez-Gutiérrez, Javier Díez-González (2025)

<a href='http://arxiv.org/abs/2505.00105'><img src='https://img.shields.io/badge/ArXiv-2505.00105-red'></a>

[📝 Paper on arXiv](https://arxiv.org/abs/2505.00105)

[📄 PDF](https://arxiv.org/pdf/2505.00105.pdf)

# Updates

[30/04/2025] Preprint available on [arXiv](https://arxiv.org/abs/2505.00105)

[??/05/2025] Code available on Github

[??/05/2025] Preprint available in [Papers with code](https://example.com)


## Abstract
Retrieval-Augmented Generation enhances language models by retrieving relevant information from external knowledge bases, relying on high-dimensional vector embeddings typically stored in float32 precision. However, storing these embeddings at scale presents significant memory challenges.

This work investigates two complementary optimization strategies on the MTEB benchmark:

1. **Quantization**: evaluating standard formats (float16, int8, binary) and low-bit floating-point types (float8)
2. **Dimensionality reduction**: assessing methods like PCA, Kernel PCA, UMAP, Random Projections and Autoencoders

### Key Findings
- **float8** achieves 4× storage reduction with minimal performance degradation (<0.3%), outperforming int8
- **PCA** emerges as the most effective dimensionality reduction technique
- Combining moderate PCA (50% dimensions) with float8 offers 8× total compression with less impact than int8 alone

---

## Implementation
Python repository to reproduce the experiments:

### Features
- Quantization types: FLOAT32, FLOAT16, BFLOAT16, FLOAT8 (E4M3, E5M2), FLOAT4 (E2M1), INT8, and Binary
- Multiple dimensionality reduction techniques with configurable parameters
- Caching system using Qdrant vector database
- Comprehensive evaluation using [MTEB](https://github.com/embeddings-benchmark/mteb) benchmarks
- Automated experiment configuration through YAML files

## Requirements
- Python 3.12
- Docker and Docker Compose (for persistent cache of embeddings)
- CUDA-capable GPU (recommended)



## Usage

1. Configure your experiments in a YAML file (see `experiments` folder examples):
   - Select model
   - Choose benchmarks
   - Define compression experiments

2. Run experiments:
```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 python run_experiments.py --config configs/experiment.yml
```

With the default configuration, the experiment uses an in-memory cache to avoid re-computing unnecesary embeddings. If you wish to use a persistent cache, spin up a qDrant instance via Docker:

```bash
docker compose up -d

HF_DATASETS_TRUST_REMOTE_CODE=1 python run_experiments.py --config configs/experiment.yml --cache-location localhost:6333
```

You can use `download_datasets.py` to download the required datasets from HuggingFace and avoid download times when running the evaluation script:

```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 python download_datasets.py --config configs/experiment.yml
```

## Citation

```bibtex
@misc{huergapérez2025optimizationembeddingsstoragerag,
      title={Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques}, 
      author={Naamán Huerga-Pérez and Rubén Álvarez and Rubén Ferrero-Guillén and Alberto Martínez-Gutiérrez and Javier Díez-González},
      year={2025},
      eprint={2505.00105},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.00105}, 
}
```

## License
MIT


