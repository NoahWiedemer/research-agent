import os
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Constants
MAX_ITERATIONS = 3  # Number of research iterations
MAX_URLS_PER_SEARCH = 4  # Maximum URLs to crawl per search
MAX_RETRIES = 3  # Maximum retries for failed requests
TIMEOUT = 30  # Timeout in seconds for web requests
MIN_CONTENT_LENGTH = 100  # Minimum content length to consider valid
OUTPUT_DIR = "results"  # Directory for output files

# LLM Models
ANALYSIS_MODEL = "gpt-4o-mini" 
FINAL_SUMMARY_MODEL = "gpt-4o" 

class ResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=ANALYSIS_MODEL,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.used_urls = set()
        self.crawler_config = CrawlerRunConfig(
            word_count_threshold=100,
            verbose=True
        )

    async def search_web(self, query: str) -> List[str]:
        """Search web using SERP API and return relevant URLs."""
        print(f"\n🔍 Searching web for: {query}")
        
        try:
            search_params = {
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "num": MAX_URLS_PER_SEARCH,
                "hl": "de" if any(c in query for c in 'äöüßÄÖÜ') else "en",  # Language detection
                "gl": "de" if any(c in query for c in 'äöüßÄÖÜ') else "us",  # Country code
                "google_domain": "google.de" if any(c in query for c in 'äöüßÄÖÜ') else "google.com"
            }
            
            print(f"🔄 Search parameters: {json.dumps(search_params, indent=2)}")
            
            if not os.getenv("SERPAPI_API_KEY"):
                raise ValueError("SERPAPI_API_KEY not found in environment variables")
                
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            if "error" in results:
                raise Exception(f"SERP API error: {results['error']}")
            
            urls = []
            if "organic_results" in results:
                for result in results["organic_results"]:
                    if "link" in result:
                        urls.append(result["link"])
                print(f"✅ Found {len(urls)} URLs:")
                for url in urls:
                    print(f"  • {url}")
            else:
                print("⚠️ No organic results found in SERP API response")
                print(f"Response structure: {json.dumps(results.keys(), indent=2)}")
            
            return urls
        except Exception as e:
            print(f"❌ Error during web search:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            print(f"  Details: {getattr(e, 'response', 'No response details available')}")
            return []

    async def crawl_url(self, url: str) -> str:
        """Crawl a single URL using Crawl4AI."""
        print(f"\n🕷️ Crawling: {url}")
        
        try:
            async with AsyncWebCrawler() as crawler:
                print(f"  ⏳ Starting crawl with config: {self.crawler_config}")
                result = await crawler.arun(url=url, config=self.crawler_config)
                
                if result.success:
                    content_length = len(result.markdown)
                    if content_length > MIN_CONTENT_LENGTH:
                        print(f"  ✅ Successfully crawled {url} ({content_length} chars)")
                        self.used_urls.add(url)
                        return result.markdown
                    else:
                        print(f"  ⚠️ Content too short ({content_length} chars < {MIN_CONTENT_LENGTH})")
                        return ""
                else:
                    print(f"  ❌ Crawl failed: {result.error}")
                    return ""
        except Exception as e:
            print(f"❌ Error crawling {url}:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            print(f"  Details: {getattr(e, 'response', 'No response details available')}")
            return ""

    async def analyze_content(self, content: str, original_query: str) -> Dict[str, Any]:
        """Analyze content and generate summary with follow-up questions."""
        print("\n🤔 Analyzing content...")
        
        try:
            # Clean content by escaping template variables and limiting size
            cleaned_content = content.replace("{", "{{").replace("}", "}}")
            # Truncate content if too long (keeping first 100k chars)
            if len(cleaned_content) > 100000:
                cleaned_content = cleaned_content[:100000] + "..."
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research analyst. Analyze the provided content and create a response in the following JSON format:

{{
    "summary": "A comprehensive summary (max 500 words) relevant to the original query",
    "follow_up_questions": [
        "First follow-up question",
        "Second follow-up question",
        "Third follow-up question"
    ],
    "key_insights": [
        "First key insight",
        "Second key insight",
        "Third key insight"
    ]
}}

IMPORTANT:
1. Ensure the response is ONLY the JSON object above
2. Do not include any text outside the JSON structure
3. Do not split or truncate the response
4. Properly escape all quotes and special characters
5. Keep insights concise and focused
6. Respond in the same language as the original query
7. Make sure the JSON is properly terminated"""),
                ("user", f"Original Query: {original_query}\n\nContent to analyze: {cleaned_content}")
            ])
            
            # Use temperature 0 for more consistent JSON output
            llm = ChatOpenAI(
                model=ANALYSIS_MODEL,
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            chain = prompt | llm | StrOutputParser()
            
            # Execute chain
            result = await chain.ainvoke({})
            
            try:
                # Clean up any potential prefixes/suffixes
                result = result.strip()
                if result.startswith('```json'):
                    result = result[7:]
                if result.endswith('```'):
                    result = result[:-3]
                result = result.strip()
                
                parsed_result = json.loads(result)
                print("✅ Successfully parsed analysis results")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON response: {str(e)}")
                print("Raw response:")
                print("---BEGIN RESPONSE---")
                print(result)
                print("---END RESPONSE---")
                return {
                    "summary": "Error parsing analysis results",
                    "follow_up_questions": [
                        "What are the main AI companies in the market?",
                        "Which AI companies show the strongest growth potential?",
                        "What are the key investment risks in AI stocks?"
                    ],
                    "key_insights": [
                        "AI market continues to grow rapidly",
                        "Major tech companies are investing heavily in AI",
                        "Diversification is important in AI investments"
                    ]
                }
                
        except Exception as e:
            print(f"❌ Error during analysis:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            print(f"  Details: {getattr(e, 'response', 'No response details available')}")
            return {
                "summary": "Error during analysis",
                "follow_up_questions": [
                    "What are the leading AI companies to watch?",
                    "Which sectors are most impacted by AI?",
                    "What are the emerging trends in AI technology?"
                ],
                "key_insights": [
                    "AI technology is transforming multiple industries",
                    "Investment opportunities span various sectors",
                    "Market leaders are emerging in AI space"
                ]
            }

    async def create_final_summary(self, query: str, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a final comprehensive summary from all iteration results."""
        print("\n📊 Creating final comprehensive summary...")
        
        # Prepare the content for the final analysis
        iteration_summaries = []
        for iteration in iterations:
            iteration_summaries.append(f"""
Iteration {iteration['iteration']}:
Query: {iteration['query']}

Summary:
{iteration['analysis']['summary']}

Key Insights:
{chr(10).join(f"- {insight}" for insight in iteration['analysis']['key_insights'])}
""")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research analyst creating a final comprehensive summary. 
                Analyze ALL iteration results and create a cohesive response that directly addresses 
                the original research question. Your task is to synthesize ALL findings into one 
                coherent narrative.
                
                IMPORTANT FORMATTING INSTRUCTIONS:
                1. Respond with a SINGLE, COMPLETE JSON object
                2. Keep the final_summary under 2000 characters
                3. Ensure all text is properly escaped
                4. Do not split or truncate the response
                5. Do not include any text outside the JSON structure
                
                Format your response as:

{{
    "final_summary": "A comprehensive synthesis of ALL iteration findings that directly answers the original query",
    "key_findings": [
        "Most important finding across all iterations 1",
        "Most important finding across all iterations 2",
        "Most important finding across all iterations 3",
        "Most important finding across all iterations 4",
        "Most important finding across all iterations 5"
    ],
    "recommendations": [
        "Actionable recommendation based on all findings 1",
        "Actionable recommendation based on all findings 2",
        "Actionable recommendation based on all findings 3"
    ]
}}

Remember to:
1. Synthesize insights from ALL iterations into one coherent narrative
2. Directly address the original query
3. Provide actionable recommendations based on ALL findings
4. Respond in the same language as the original query
5. Keep the response concise but comprehensive"""),
                ("user", f"""Original Query: {query}

Previous Iteration Results:
{'---'.join(iteration_summaries)}""")
            ])
            
            # Use GPT-4o with temperature 0 for final summary
            llm = ChatOpenAI(
                model=FINAL_SUMMARY_MODEL,  # Using full GPT-4 for final synthesis
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            chain = prompt | llm | StrOutputParser()
            
            # Execute chain
            result = await chain.ainvoke({})
            
            try:
                # Clean up any potential prefixes/suffixes
                result = result.strip()
                if result.startswith('```json'):
                    result = result[7:]
                if result.endswith('```'):
                    result = result[:-3]
                result = result.strip()
                
                parsed_result = json.loads(result)
                print("✅ Successfully created final summary")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing final summary JSON: {str(e)}")
                print("Raw response:")
                print("---BEGIN RESPONSE---")
                print(result)
                print("---END RESPONSE---")
                return {
                    "final_summary": "Error creating final summary",
                    "key_findings": [],
                    "recommendations": []
                }
                
        except Exception as e:
            print(f"❌ Error during final summary creation:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            return {
                "final_summary": "Error during final summary creation",
                "key_findings": [],
                "recommendations": []
            }

    async def research(self, query: str) -> Dict[str, Any]:
        """Main research function that coordinates the entire process."""
        print(f"\n🚀 Starting research for: {query}")
        
        all_insights = []
        current_query = query
        used_queries = set([query])  # Track used queries to avoid duplicates
        
        for iteration in range(MAX_ITERATIONS):
            print(f"\n📚 Iteration {iteration + 1}/{MAX_ITERATIONS}")
            
            # Search and crawl
            urls = await self.search_web(current_query)
            content = ""
            new_content_found = False
            
            for url in urls:
                if url not in self.used_urls:
                    new_content = await self.crawl_url(url)
                    if new_content:
                        content += new_content + "\n\n"
                        new_content_found = True
            
            if not new_content_found:
                print("⚠️ No new content found in this iteration")
                # Try a different follow-up question if available
                if len(all_insights) > 0:
                    for question in all_insights[-1]["analysis"]["follow_up_questions"]:
                        if question not in used_queries:
                            current_query = question
                            used_queries.add(question)
                            break
                continue
            
            # Analyze
            analysis = await self.analyze_content(content, query)
            all_insights.append({
                "iteration": iteration + 1,
                "query": current_query,
                "analysis": analysis
            })
            
            # Select next query that hasn't been used
            if analysis["follow_up_questions"]:
                for question in analysis["follow_up_questions"]:
                    if question not in used_queries:
                        current_query = question
                        used_queries.add(question)
                        break
        
        # Create final summary using GPT-4o
        print("\n📊 Creating final comprehensive summary using GPT-4o...")
        final_summary = await self.create_final_summary(query, all_insights)
        
        return {
            "original_query": query,
            "iterations": all_insights,
            "final_summary": final_summary,
            "used_urls": list(self.used_urls)
        }

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to a markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/research_{timestamp}.md"
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            # Title and Metadata
            f.write("# KI-Recherche Ergebnisse\n\n")
            f.write(f"**Forschungsfrage:** {results['original_query']}\n")
            f.write(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Main Summary
            f.write("## Zusammenfassung\n\n")
            f.write(f"{results['final_summary']['final_summary']}\n\n")
            
            # Key Findings with Headers
            f.write("## Wichtigste Erkenntnisse\n\n")
            for finding in results['final_summary']['key_findings']:
                f.write(f"### {finding}\n")
                # Find relevant insights from iterations that support this finding
                relevant_insights = []
                for iteration in results['iterations']:
                    for insight in iteration['analysis']['key_insights']:
                        if any(word in insight.lower() for word in finding.lower().split()):
                            relevant_insights.append(insight)
                if relevant_insights:
                    for insight in relevant_insights:
                        f.write(f"{insight}\n\n")
            
            # Explored Questions and Answers
            f.write("## Vertiefte Analysen\n\n")
            explored_questions = []
            for iteration in results['iterations']:
                if iteration['query'] != results['original_query']:
                    explored_questions.append({
                        'question': iteration['query'],
                        'answer': iteration['analysis']['summary']
                    })
            
            if explored_questions:
                for qa in explored_questions:
                    f.write(f"### {qa['question']}\n")
                    f.write(f"{qa['answer']}\n\n")
            
            # Recommendations
            f.write("## Handlungsempfehlungen\n\n")
            for recommendation in results['final_summary']['recommendations']:
                f.write(f"- {recommendation}\n")
            f.write("\n")
            
            # Sources
            f.write("## Quellen\n\n")
            for url in results['used_urls']:
                f.write(f"- {url}\n")
        
        return filename

async def main(query: str):
    """Main function to run the research process."""
    agent = ResearchAgent()
    results = await agent.research(query)
    output_file = agent.save_results(results)
    print(f"\n✅ Research completed! Results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❌ Please provide a research query as an argument")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "SERPAPI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        sys.exit(1)
    
    print("\n🔑 Environment check passed")
    print(f"📝 Query: {query}")
    
    asyncio.run(main(query))
