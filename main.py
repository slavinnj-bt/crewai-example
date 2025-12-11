#!/usr/bin/env python3
import os
from dotenv import load_dotenv
# Need the SpanProcessor to automatically log, batch and export traces
from braintrust.otel import BraintrustSpanProcessor
# OTEL API 
from opentelemetry import trace
# OTEL auto instrumentation
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
# Factory for Tracer configs
from opentelemetry.sdk.trace import TracerProvider
from crewai import Agent, Crew, Task, Process, LLM


def setup_tracing(project_name: str = "SlavinScratchArea", experiment_name: str = "crewai-demo") -> None:
    """Initialize Braintrust tracing for CrewAI and OpenAI"""
    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        provider = current_provider
    else:
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    parent = f"project_name:{project_name}"
    # Init the span processor and attach to the project
    provider.add_span_processor(BraintrustSpanProcessor(parent=parent))
    CrewAIInstrumentor().instrument(tracer_provider=provider)
    OpenAIInstrumentor().instrument(tracer_provider=provider)


def main():
    # TODO: add this as user input
    thesis = "small-cap Chinese equities"
    """Main execution function"""
    # Load environment variables
    load_dotenv()

    # Verify required env vars
    # OPENAI required for Crew, BRAINTRUST needed for submitting traces
    required_vars = ["BRAINTRUST_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Initialize tracing
    setup_tracing()

    # Create LLM instance
    llm = LLM(model="gpt-4o-mini")

    # Stock researcher
    researcher = Agent(
        role="Equity Research Analyst",
        goal="Identify stocks that align with an investment theme",
        backstory="You are a skilled equity analyst that picks stocks that conform to a specific investment theme."
            "You consider the success of the stock based on fundamental and technical analysis, and how well the stock fits "
            "the desired investment theme.",
        llm=llm,
        inject_date=True,
        respect_context_window=True,
        date_format="%B %d, %Y",  # Format as "May 21, 2025"
        verbose=True
    )

    # Portfolio builder
    writer = Agent(
        role="Portfolio Constructor",
        goal="Construct a portfolio for an ETF ",
        backstory="You are a skilled advisor that constructs portfolios based on a given theme and equity research."
            "You are also an expert writer that can write a concise summary of the fund's investment thesis and describe its objective and composition to a prospective investor.",
        llm=llm,
        inject_date=True,
        date_format="%B %d, %Y",  # Format as "May 21, 2025"
        verbose=True
    )

    # Define tasks
    research_task = Task(
        description=f"Research stocks that fit the investment thesis: {thesis}",
        expected_output="A detailed report on stocks that match the given thesis. You will return a short description of the equity, why it was included, and relevant news/developments.",
        agent=researcher
    )

    writing_task = Task(
        description="Create a list of 20 stocks for an ETF bucket based on the research results. Attempt to pick the most successful stocks. You will try to diversify the list.",
        expected_output="A list of 20 stocks ",
        agent=writer,
        context=[research_task] # wait for research before building ETF portfolio
    )

    # Create and run the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff()

    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(result)


if __name__ == "__main__":
    main()
