# LLMDocParser

A package for parsing PDFs and analyzing their content using LLMs.

This package is an improvement based on the concept of [gptpdf](https://github.com/CosmosShadow/gptpdf/tree/main).

## Method
gptpdf uses PyMuPDF to parse PDFs, identifying both text and non-text regions. It then merges or filters the text regions based on certain rules, and inputs the final results into a multimodal model for parsing. This method is particularly effective.

Based on this concept, I made some minor improvements.

### Main Process
Using a layout analysis model, each page of the PDF is parsed to identify the type of each region, which includes Text, Title, Figure, Figure caption, Table, Table caption, Header, Footer, Reference, and Equation. The coordinates of each region are also obtained.

Layout Analysis Result Example:
```
[{'header': ((101, 66, 436, 102), 0)},
 {'header': ((1038, 81, 1088, 95), 1)},
 {'title': ((106, 215, 947, 284), 2)},
 {'text': ((101, 319, 835, 390), 3)},
 {'text': ((100, 565, 579, 933), 4)},
 {'text': ((100, 967, 573, 1025), 5)},
 {'text': ((121, 1055, 276, 1091), 6)},
 {'reference': ((101, 1124, 562, 1429), 7)},
 {'text': ((610, 565, 1089, 930), 8)},
 {'text': ((613, 976, 1006, 1045), 9)},
 {'title': ((612, 1114, 726, 1129), 10)},
 {'text': ((611, 1165, 1089, 1431), 11)},
 {'title': ((1011, 1471, 1084, 1492), 12)}]
```
This result includes the type, coordinates, and reading order of each region. By using this result, more precise rules can be set to parse the PDF.

Finally, input the images of the corresponding regions into a multimodal model, such as GPT-4o or Qwen-VL, to directly obtain text blocks that are friendly to RAG solutions.

| filepath                                  | type            | page_no | filename                  | content               |
|-------------------------------------------|-----------------|---------|---------------------------|-----------------------|
| output/page_1_title.png                   | Title           | 1       | attention is all you need | [Text Block 1]        |
| output/page_1_text.png                    | Text            | 1       | attention is all you need | [Text Block 2]        |
| output/page_2_figure.png                  | Figure          | 2       | attention is all you need | [Text Block 3]        |
| output/page_2_figure_caption.png          | Figure caption  | 2       | attention is all you need | [Text Block 4]        |
| output/page_3_table.png                   | Table           | 3       | attention is all you need | [Text Block 5]        |
| output/page_3_table_caption.png           | Table caption   | 3       | attention is all you need | [Text Block 6]        |
| output/page_1_header.png                  | Header          | 1       | attention is all you need | [Text Block 7]        |
| output/page_2_footer.png                  | Footer          | 2       | attention is all you need | [Text Block 8]        |
| output/page_3_reference.png               | Reference       | 3       | attention is all you need | [Text Block 9]        |
| output/page_1_equation.png                | Equation        | 1       | attention is all you need | [Text Block 10]       |

See more in llm_parser.py main function.

## Installation

```commandline
pip install llmdocparser
```

## Usage

```python
from llmdocparser.llm_parser import get_image_content

content = get_image_content(
    llm_type="azure",
    pdf_path="path/to/your/pdf",
    output_dir="path/to/output/directory",
    max_concurrency=5,
    azure_deployment="azure-gpt-4o",
    azure_endpoint="your_azure_endpoint",
    api_key="your_api_key",
    api_version="your_api_version"
)
print(content)
```

**Parameters**

* llm_type: str
  
  The options are azure, openai, dashscope.
* pdf_path: str
  
  Path to the PDF file.
* output_dir: str
  
  Output directory to store all parsed images.

* max_concurrency: int
  
  Number of GPT parsing worker threads. Batch calling details: [Batch Support](https://python.langchain.com/v0.2/docs/integrations/llms/#features-natively-supported)

If using Azure, the azure_deployment and azure_endpoint parameters need to be passed; otherwise, only the API key needs to be provided.

* base_url: str
  
  OpenAI Compatible Server url. Detail: [OpenAI-Compatible Server](https://python.langchain.com/v0.2/docs/integrations/llms/vllm/#openai-compatible-server)

## Cost

Using the 'Attention Is All You Need' paper for analysis, the model chosen is GPT-4o, costing as follows:
```
Total Tokens: 44063
Prompt Tokens: 33812
Completion Tokens: 10251
Total Cost (USD): $0.322825
```
Average cost per page: $0.0215


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lazyFrogLOL/llmdocparser&type=Date)](https://star-history.com/#lazyFrogLOL/llmdocparser&Date)