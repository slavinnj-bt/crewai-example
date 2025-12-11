# CrewAI with Braintrust Tracing - Quick Start

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   - `BRAINTRUST_API_KEY`: Your Braintrust API key
   - `BRAINTRUST_PARENT`: Format is `project_name:<your_project_name>`
   - `OPENAI_API_KEY`: Your OpenAI API key

## Run

```bash
python main.py
```
