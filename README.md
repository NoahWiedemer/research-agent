# Research Agent 47

An AI-powered research assistant that performs automated web research, analysis, and summary generation using GPT-4 and SERP API.

## Features

- 🔍 Automated web searches using SERP API
- 🕷️ Web content crawling and extraction
- 🤖 Multi-iteration research with follow-up questions
- 📊 GPT-4 powered content analysis and summarization
- 📝 Markdown report generation with structured insights

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```


## Usage

Run the script with your research query:
```bash
python agent_47.py "What are the latest developments in quantum computing?"
```

The script will:
1. Perform multiple iterations of web research
2. Analyze and summarize findings
3. Generate follow-up questions
4. Create a comprehensive markdown report in the `results` directory

## Output

Results are saved as markdown files in the `results` directory with:
- Executive summary
- Key findings
- Detailed analysis
- Recommendations
- Source URLs

## Requirements

- Python 3.8+
- OpenAI API key
- SERP API key