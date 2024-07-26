# LLMDocParser

A package for parsing PDFs and analyzing their content using LLMs.

## Installation

```commandline
pip install llmdocparser
```

## Usage

```python
from llmdocparser import get_image_content

content = get_image_content(
    llm_type="azure",
    pdf_path="path/to/your/pdf",
    output_dir="path/to/output/directory",
    max_concurrency=5,
    azure_deployment="azure-gpt-4o",
    azure_endpoint="your_azure_endpoint",
    api_key="your_api_key"
)
print(content)