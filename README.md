# FinSight AI
## Intelligent Financial Analytics & Document Intelligence Platform

FinSight AI is a machine learning and financial document intelligence project combining customer churn analytics, churn prediction, model explainability, and evidence-grounded financial report question answering.

## Features

- Interactive customer analytics dashboard
- Customer churn analysis and prediction
- SHAP-based model explainability
- Financial PDF question answering
- Hybrid semantic and keyword retrieval
- Reporting-year validation to help prevent unsupported answers
- Retrieved evidence with PDF page references

## Technology Stack

- Python, Pandas, NumPy
- Scikit-learn and SHAP
- Streamlit
- Sentence Transformers
- Groq API
- Joblib
- PyPDF

## Project Structure

```text
FinSightAI/
├── app.py
├── document_app.py
├── document_intelligence.py
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── finsight_churn_model.joblib
└── data/
    └── processed/
        └── telco_churn_cleaned.csv
```

## How Financial Document Intelligence Works

1. Upload a financial PDF.
2. Extract text from its pages and divide it into chunks.
3. Generate sentence embeddings.
4. Retrieve relevant passages using semantic and keyword-based signals.
5. Validate the requested reporting year against the document.
6. Generate an answer using the retrieved evidence.
7. Display supporting PDF page references.

## Selected Validation Results

Selected tests using Apple FY2025 and Microsoft FY2025 annual reports returned:

| Test | Expected reported value |
|---|---:|
| Apple FY2025 Services net sales | $109,158 million |
| Apple FY2025 net income | $112,010 million |
| Microsoft FY2025 operating income | $128,528 million |
| Microsoft FY2025 net income | $101,832 million |
| Microsoft FY2024 operating income | $109,433 million |

A request for Microsoft's FY2030 operating income was correctly refused because that reporting-year figure was not supported by the uploaded document.

These are selected test cases, not a guarantee of accuracy across all reports or questions.

## Setup

Use Python 3.10 or a compatible Python version supported by the listed dependencies.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key as an environment variable named `GROQ_API_KEY`. In Google Colab, you can use Colab Secrets. Never put the actual key in source code or commit it to GitHub.

### Run the main dashboard

```bash
streamlit run app.py --server.port 8501
```

### Run financial document intelligence

```bash
streamlit run document_app.py --server.port 8502
```

Run each application in a separate terminal or process. Both applications may require access to the project's model and data files.

## Security

- Never commit API keys, credentials, or private documents.
- Keep local secrets and environment files out of version control.
- Review files before publishing the repository.

## Limitations

- PDF text extraction quality can affect retrieval and answer accuracy.
- Complex financial tables may be difficult to parse.
- Answers should be checked against the original report before important decisions.
- API access depends on available provider credits and service availability.
- Current validation covers selected questions rather than every possible case.

## Future Improvements

- Automated regression tests
- Broader annual-report evaluation
- Better financial table extraction
- Scanned PDF support through OCR
- Retrieval and answer-quality evaluation
- Reproducible dependency and deployment configuration

## Project Status

Core dashboard pages and selected financial document question-answering cases have been tested. Further automated testing is recommended before production use.
