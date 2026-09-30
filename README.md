# Agentic Data Intelligence

> An evidence-driven, local-first data intelligence platform for profiling, understanding, validating, cleaning, mapping, and loading structured and unstructured data.

Agentic Data Intelligence is a production-oriented data engineering and AI system designed to turn unfamiliar datasets and documents into validated, analysis-ready data.

Instead of using an LLM as a generic chatbot, the project combines deterministic data engineering with bounded Agentic AI. Python performs exact calculations, validation, transformation, and database operations, while a local language model is used only where semantic reasoning adds value.

The result is a system that can ingest heterogeneous files, understand their structure, generate evidence-backed data quality rules, quarantine invalid records, produce clean datasets, suggest source-to-target mappings, and safely load approved data into PostgreSQL.

---

## Why This Project Exists

Real-world data pipelines rarely receive perfectly structured input.

Incoming data may contain:

- inconsistent schemas
- malformed dates
- incorrect numeric values
- broken email addresses
- unexpected data types
- duplicate records
- missing business context
- semi-structured documents
- unclear source-to-target mappings

Traditional ETL pipelines usually solve these problems with manually written rules for every source.

Pure LLM-based systems have the opposite problem: they may infer rules that are not actually supported by the source data or business requirements.

Agentic Data Intelligence takes a hybrid approach:

```text
Deterministic Engineering
        +
Semantic AI Reasoning
        +
Evidence-Based Guardrails
        =
Reliable Data Intelligence
```

The AI can reason about data, but it cannot arbitrarily execute Python, SQL, or unvalidated business rules.

---

# Core Capabilities

## Universal Data Ingestion

The ingestion layer converts supported inputs into a common internal `DataAsset` representation.

Supported formats include:

| Category | Formats |
|---|---|
| Delimited | CSV, TSV |
| Spreadsheet | XLSX, XLSM, XLS |
| Semi-structured | JSON, JSONL, NDJSON, XML |
| Columnar | Parquet |
| Text | TXT, LOG, Markdown |
| Documents | PDF, DOCX |

The ingestion layer also preserves lexically significant values such as:

```text
+917000000001
001245
000007
12345678901234567890
```

rather than allowing automatic numeric inference to silently modify them.

---

## Automated Data Profiling

The deterministic profiler analyzes datasets without depending on the language model.

It evaluates characteristics such as:

```text
row counts
null percentages
distinct values
duplicate rows
numeric statistics
text statistics
datetime statistics
pattern matches
outlier evidence
column samples
inferred data types
```

Pattern detection includes structures such as:

```text
email
URL
UUID
IP address
integer-like text
decimal-like text
date hints
```

Statistical analysis includes both IQR and MAD-based outlier evidence.

Outliers are treated as evidence only. They are not automatically classified as invalid data.

---

## Semantic Dataset Understanding

A local language model analyzes profiling evidence and assigns semantic meaning to columns.

Examples include:

```text
identifier
identifier_candidate
code
numeric_measure
numeric_value
datetime
date
email
url
ip_address
boolean
category
free_text
name
address
phone
currency
percentage
unknown
```

The semantic layer is intentionally constrained.

It is not allowed to invent tables, columns, or unsupported business requirements.

---

# Agentic Analysis

The system uses a bounded agent loop rather than a single prompt-response workflow.

```text
Observe
   ↓
Reason
   ↓
Request Allowed Tool
   ↓
Execute Deterministically
   ↓
Observe Result
   ↓
Continue or Finalize
```

The LLM cannot directly execute arbitrary code.

Instead, it can request only predefined analysis tools.

Available tools include:

```text
list_tables
table_overview
sample_rows
column_distribution
missingness_analysis
duplicate_analysis
uniqueness_analysis
correlation_analysis
outlier_analysis
column_pair_analysis
date_range_analysis
document_excerpt
```

Python validates and executes every request.

Repeated calls and uncontrolled agent loops are bounded to prevent unnecessary execution.

---

# Dynamic Data Quality Rules

The platform generates data quality rules from deterministic profiling evidence and semantic interpretation.

Supported rule types include:

```text
type_check
pattern_match
not_null
unique
range
allowed_values
length
datetime_parse
not_future
duplicate_rows
```

Example:

```text
Column semantic type: email
            ↓
Generated structural rule:
pattern_match(email)
            ↓
Python executes validation
            ↓
Malformed records are identified
```

Another example:

```text
Column semantic type: name
Semantic confidence >= threshold
            ↓
Structural contains_letters rule
            ↓
"Rahul Sharma"      → valid
"José García"       → valid
"李明"               → valid
"900113"            → invalid
```

Name validation deliberately avoids restrictive English-only regular expressions.

---

# Evidence-Based Rule Governance

Not every observed pattern should become a business rule.

For example, these assumptions are intentionally rejected:

```text
0 nulls observed
≠
column must always be NOT NULL
```

```text
100% unique values observed
≠
column is guaranteed to be unique
```

```text
maximum observed value = 100
≠
business maximum is 100
```

Rules representing business requirements require explicit supporting evidence.

This applies to rules such as:

```text
not_null
unique
range
allowed_values
length
not_future
duplicate_rows
```

This prevents the AI layer from turning temporary observations into permanent business constraints.

---

# Row-Level Data Quarantine

Data quality validation is connected directly to the pipeline execution path.

Blocking failures result in row-level partitioning:

```text
Source Dataset
      |
      v
Data Quality Rules
      |
      +----------------------+
      |                      |
      v                      v
 Valid Rows             Rejected Rows
      |                      |
      v                      v
Clean Dataset           DQ Metadata
      |
      v
Mapping / Transformation
      |
      v
Optional Database Load
```

Rejected records retain:

```text
_dq_source_row_position
_dq_failed_rule_count
_dq_failed_rule_ids
_dq_rejection_reasons
```

This provides traceability instead of silently deleting invalid records.

---

# Clean Dataset Generation

The pipeline automatically writes DQ-valid records to:

```text
reports/clean_data/
```

Example:

```text
reports/
└── clean_data/
    └── customer_data_clean.csv
```

The clean-data path follows an important principle:

> Valid records are preserved rather than silently repaired.

The platform does not invent replacement values or modify valid source data simply to make a dataset appear cleaner.

---

# Source-to-Target Mapping

The mapping engine compares source columns with PostgreSQL target schemas.

Mapping confidence is calculated using multiple signals, including:

```text
column-name similarity
data-type compatibility
semantic compatibility
table context
```

Mappings can be classified as:

```text
automatic
manual review required
```

For ambiguous mappings, the local model can reason only over candidates already generated by Python.

It cannot invent arbitrary destination columns.

Example:

```text
abc_code
    ↓
quality_records.abc_code

measurement
    ↓
quality_records.measurement_value

event_time
    ↓
quality_records.event_time

contact
    ↓
quality_records.contact_email
```

Mappings requiring review must be explicitly approved before database loading.

---

# Safe Transformation

Transformations are deterministic and mapping-driven.

Supported behavior includes:

```text
direct column mapping
safe type casting
target schema validation
required-column validation
mapping conflict detection
```

The transformation stage receives only DQ-valid records.

Rejected records are quarantined before the database load path.

---

# PostgreSQL Integration

The database layer uses:

```text
SQLAlchemy
psycopg
PostgreSQL
```

Database loading is intentionally opt-in.

Analysis alone does not write data to PostgreSQL.

Before loading, the system validates:

```text
target table existence
target schema
mapped destination columns
required fields
approved mappings
transformed data
```

Database operations use transactional inserts and rollback behavior.

The application does not perform destructive operations such as:

```text
DELETE
TRUNCATE
REPLACE
arbitrary AI-generated SQL
```

during normal pipeline execution.

---

# Unstructured Document Processing

PDF and DOCX files can be processed as document sources.

The document ingestion layer extracts:

```text
document text
detectable PDF tables
DOCX tables
structured evidence
```

A guarded document extraction stage can then identify structured information while preserving the source document for provenance and auditability.

The system does not automatically assume that prose contains a database table.

---

# Streamlit Application

The project includes an interactive Streamlit interface for running the pipeline without using the command line.

The application provides dedicated views for:

```text
Data Profiler
Agentic Analysis
Data Quality
Rejected Rows
Clean Dataset
Source → Target
Report
```

Database loading remains explicitly controlled even from the UI.

---

# System Architecture

```text
                    ┌───────────────────────────┐
                    │        Input Source       │
                    │ CSV / JSON / PDF / DOCX   │
                    │ Excel / XML / Parquet ... │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     Universal Ingestion   │
                    │         DataAsset         │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Deterministic Data Profiler│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │    Semantic Analyzer      │
                    │    Local Ollama / Qwen    │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      Agentic Analysis     │
                    │  Bounded Tool Execution   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Dynamic DQ Generator    │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Deterministic Rule Engine │
                    └─────────────┬─────────────┘
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                       ▼                     ▼
              ┌────────────────┐    ┌────────────────┐
              │   Valid Rows   │    │ Rejected Rows  │
              └───────┬────────┘    │ + DQ Metadata  │
                      │             └────────────────┘
                      ▼
              ┌────────────────┐
              │ Clean Dataset  │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Source → Target│
              │     Mapper     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Transformation │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │   PostgreSQL   │
              │ Explicit Opt-in│
              └────────────────┘
```

---

# Project Structure

```text
agentic-data-intelligence/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── config.py
│   ├── ingestion.py
│   ├── profiler.py
│   ├── analysis_tools.py
│   ├── llm_client.py
│   ├── semantic_analyzer.py
│   ├── rule_generator.py
│   ├── rule_executor.py
│   ├── target_schema.py
│   ├── mapper.py
│   ├── transformer.py
│   ├── database.py
│   ├── agent.py
│   ├── document_extractor.py
│   ├── report.py
│   └── main.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_profiler.py
│   ├── test_rules.py
│   └── test_mapper.py
│
├── samples/
├── reports/
│   └── clean_data/
│
├── config/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data Processing | pandas, NumPy |
| Validation Models | Pydantic |
| Local AI | Ollama |
| LLM | Qwen |
| Database | PostgreSQL |
| Database Access | SQLAlchemy, psycopg |
| PDF Processing | PyMuPDF |
| Word Processing | python-docx |
| Excel | openpyxl, xlrd |
| Parquet | PyArrow |
| XML | lxml / ElementTree |
| UI | Streamlit |
| Testing | pytest |
| Reporting | JSON, HTML |
| CLI | Python module entry point |

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd agentic-data-intelligence
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Local LLM Setup

The project uses Ollama so semantic and agentic processing can run locally.

Install Ollama and pull the configured model.

Example:

```bash
ollama pull qwen3:4b
```

Verify the model is available:

```bash
ollama list
```

The project is designed so deterministic pipeline functionality can still operate when optional AI enrichment is unavailable.

---

# Environment Configuration

Copy the example environment file:

### Windows

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Configure `.env` according to your local environment.

Use `.env.example` as the source of truth for supported configuration options.

Database loading can remain disabled when PostgreSQL integration is not required.

---

# Running the Pipeline

Basic CLI execution:

```bash
python -m src.main data.csv
```

Example with bounded agent rounds:

```bash
python -m src.main data.csv --max-agent-rounds 2
```

The pipeline produces assessment files under:

```text
reports/
```

and clean datasets under:

```text
reports/clean_data/
```

---

# Running the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser and allow datasets to be uploaded and analyzed interactively.

---

# Testing

Run the complete automated test suite with:

```bash
python -m pytest tests -v
```

Current verified project state:

```text
35 tests passed
```

The suite covers the primary ingestion, profiling, rule execution, row partitioning, rule generation, and mapping behavior.

---

# Large-File Validation

A corruption stress test was performed using a synthetic CSV containing:

```text
49,184 source rows
12 columns
approximately 9.2 MB
```

The dataset intentionally contained multiple corruption categories.

Verified result:

| Validation | Result |
|---|---:|
| Original rows | 49,184 |
| Clean rows | 46,751 |
| Rejected rows | 2,433 |
| Expected unique blocking failures | 2,433 |
| Numeric names remaining | 0 |
| Invalid ages remaining | 0 |
| Invalid signup dates remaining | 0 |
| Malformed non-empty emails remaining | 0 |
| Invalid balances remaining | 0 |
| Invalid boolean values remaining | 0 |
| Invalid `last_login` values remaining | 0 |
| Phone values preserving `+91` | 46,751 / 46,751 |
| Exact expected clean dataset match | `True` |

The final comparison confirmed:

```text
Expected shape: (46751, 12)
Actual shape:   (46751, 12)
Same columns:   True
Exact retained-data match: True
```

This validation demonstrates that the pipeline rejected the expected blocking corruption classes while leaving retained source values unchanged.

These numbers represent a local project validation run and should not be interpreted as a general performance benchmark.

---

# Intentional Non-Blocking Findings

The stress-test clean dataset still contained:

```text
225 blank email values
208 duplicate excess rows
```

This behavior is deliberate.

Blank values and duplicate rows are not universally invalid.

Automatically rejecting them would require assumptions such as:

```text
email must always be present
```

or:

```text
duplicate records are never allowed
```

Those are business requirements rather than universal structural rules.

The platform therefore requires supporting evidence before enforcing them as blocking constraints.

---

# Example Pipeline Output

```json
{
  "source": "corrupted_customer_data_10mb.csv",
  "source_type": "csv",
  "quality_score": 99.37,
  "tables_profiled": 1,
  "documents_profiled": 0,
  "agent_rounds": 1,
  "agent_tools_executed": 0,
  "dq_rules": 8,
  "valid_rows": 46751,
  "rejected_rows": 2433,
  "database_load": null
}
```

The number of agent rounds and tools executed depends on the evidence available for a particular dataset.

An agent performing zero tool calls is valid when the supplied profiling evidence is already sufficient.

---

# Design Principles

### Deterministic Where Precision Matters

Calculations, rule execution, row partitioning, transformations, mapping validation, and database writes remain deterministic.

### AI Where Reasoning Adds Value

The LLM is used for semantic interpretation, bounded investigation, and ambiguous reasoning rather than exact calculations.

### No Arbitrary Code Execution

The model cannot generate and execute arbitrary Python or SQL.

### Evidence Over Guessing

Observed data characteristics are not automatically converted into business requirements.

### Human Review for Ambiguity

Low-confidence source-to-target mappings require approval.

### Explicit Database Writes

Database loading must be intentionally enabled.

### Source Preservation

Valid source records are retained without silent repair or invented replacement values.

### Local-First AI

Ollama allows semantic reasoning to operate locally without requiring a paid hosted LLM API.

---

# Example Use Cases

The architecture is suitable for workflows such as:

```text
Unstructured File → Structured Data
Source-to-Target Mapping
Automated Data Profiling
Data Quality Assessment
Data Quality Rule Generation
Data Quarantine
Clean Dataset Generation
Schema Discovery
ETL Pre-Validation
PostgreSQL Loading
Document Intelligence
Data Migration Assessment
Data Onboarding
```

---

# What Makes This Agentic

This project is not a wrapper around an LLM API.

The AI participates inside a controlled engineering workflow:

```text
AI decides what should be investigated
        ↓
Python decides what is allowed
        ↓
Python executes the tool
        ↓
AI receives grounded evidence
        ↓
Python validates the final output
```

The model therefore acts as a reasoning component rather than an unrestricted execution engine.

---

# Reliability and Safety

The project includes several safeguards designed for data-engineering workloads:

```text
whitelisted agent tools
bounded reasoning rounds
validated structured LLM responses
deterministic rule IDs
rule-reference validation
parameter validation
business-evidence requirements
row-level rejection metadata
safe transformation logic
mapping approval controls
transactional database loading
explicit database opt-in
no arbitrary SQL execution
```

---

# Current Status

The core pipeline is implemented and operational across:

```text
Universal Ingestion
Data Profiling
Semantic Analysis
Agentic Investigation
Dynamic DQ Generation
Deterministic DQ Execution
Row Quarantine
Clean Dataset Generation
Source-to-Target Mapping
Transformation
PostgreSQL Loading
JSON / HTML Reporting
Streamlit UI
```

Core automated tests currently pass successfully, and the structured-data cleaning pipeline has been validated against a large synthetic corruption dataset.

---

# Future Extensions

Potential extensions can be added without replacing the current architecture:

```text
additional deterministic profiling methods
more domain-aware semantic types
schema version comparison
pipeline-run audit persistence
additional mapping strategies
data lineage visualization
batch source processing
REST API exposure
containerized deployment
CI/CD validation
```

These are extensions rather than requirements for the current implementation.

---

# Project Philosophy

The core principle behind Agentic Data Intelligence is simple:

> AI should reason about data, not bypass engineering controls.

Reliable data systems still need deterministic validation, traceable decisions, explicit permissions, and reproducible outputs.

This project combines those engineering principles with local Agentic AI to make unfamiliar data easier to understand and safer to operationalize.

---

## Author

**Aditya Mohan Srivastava**

Built as an end-to-end Agentic AI and Data Engineering project focused on practical enterprise data intelligence, validation, mapping, and governed automation.