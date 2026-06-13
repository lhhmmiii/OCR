# Invoice OCR Pipeline 🧾

An end-to-end Optical Character Recognition (OCR) pipeline dedicated to processing Vietnamese invoices. The system detects text, recognizes it using custom PyTorch weights, attempts layout reconstruction, and uses Google's Gemini LLM to semantically extract structured invoice details. Finally, it uses a Streamlit UI to let users visually review, edit, and save the verified invoice to a PostgreSQL database.

## 🌟 Features

- **Text Detection:** Leverages [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for robust, multi-oriented text detection.
- **Vietnamese Text Recognition:** Integrates [VietOCR](https://github.com/pbcquoc/vietocr) to handle local nuances and diacritics.
- **Layout Reconstruction:** Groups text into logical blocks and reconstructs tabular/columnar layout structures (`src/layout`).
- **Semantic Extraction (LLM):** Uses Google Gemini (`google-genai`) to ingest raw OCR text into predefined schemas (e.g., extracting Vendor Name, Total Amount, Date).
- **Interactive UI:** A highly polished, glassmorphism-styled Streamlit interface that shows annotated image previews and allows manual override of the extracted data.
- **Persistent Storage:** Saves corrected invoices to a PostgreSQL database, orchestrated seamlessly with SQLAlchemy and Alembic migrations.

## 🛠 Project Architecture & Structure

```
├── alembic/                # Alembic database migration scripts
├── alembic.ini             # Alembic configuration
├── app.py                  # Streamlit web application
├── main.py                 # Command Line Interface (CLI) entry point
├── demo.py                 # Demonstrations and experimental scripts
├── docker-compose.yml      # PostgreSQL Database deployment via Docker
├── requirements.txt        # Python dependency list
├── checkpoints/            # Pre-trained custom model weights
│   └── text_recognizer_model.pth
├── notebooks/              # Jupyter notebooks for data analysis & experiments
├── src/                    # Core pipeline logic
│   ├── layout/             # Layout algorithms and spatial grouping
│   ├── llm/                # Connecting to Gemini and parsing Pydantic schemas
│   ├── ocr/                # Text detection & VietOCR inference wrapper
│   ├── schemas/            # Data models and configs
│   └── utils/              # Helper utilities (image processing, etc.)
└── database/               # SQL ORM layer
    ├── database.py         # Engine and connection configurations
    ├── models.py           # SQLAlchemy declarative base models
    └── repository.py       # CRUD operations
```

## 📋 Prerequisites

- Python 3.9+
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose (for PostgreSQL)
- Google Gemini API Key

## 🚀 Setup & Installation

**1. Clone the repository and setup the environment:**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

**2. Database Initialization (Docker):**
We provide a `docker-compose.yml` to effortlessly spin up a Postgres database on port `5433`.
```bash
docker-compose up -d
```

**3. Environment Variables:**
Create a `.env` file in the root directory based on the following snippet:
```dotenv
# Database Configuration
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=invoices

# Gemini LLM Extraction
GEMINI_API_KEY=your_gemini_api_key_here
```

**4. Run Database Migrations:**
Ensure your database tables match the SQLAlchemy models.
```bash
alembic upgrade head
```

**5. Model Weights:**
Make sure your custom VietOCR weights (`text_recognizer_model.pth`) are placed inside the `checkpoints/` folder.

## 💻 Usage

### Streamlit Application (Recommended)
Launch the interactive web UI to upload images, view OCR box annotations, edit extraction results, and browse the database history.
```bash
streamlit run app.py
```

### Command Line Interface (CLI)
You can directly run the pipeline via CLI for debugging or headless processing. 

Run a basic full inference (detection + recognition):
```bash
python main.py --image path/to/your/invoice.jpg
```

Run with debug visualizations saved to disk:
```bash
python main.py --image path/to/your/invoice.jpg --save_vis result/annotated.jpg
```

Run specialized layout reconstruction modules:
```bash
python main.py --image invoice.jpg --layout --json-out result/layout.json
```

## 📝 License
This project is intended for personal and educational use. (Update this section with your explicit license if required).
