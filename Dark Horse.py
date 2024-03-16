from crewai import Agent , Crew , Task
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_anthropic import ChatAnthropic
from logging.handlers import RotatingFileHandler
import logging
import os
from os import getenv

# Set up rotating log file
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('my_script.log', maxBytes=1000000, backupCount=5)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

try:
    Claude_3 = os.getenv("ANTHROPIC_API_KEY") # Create .env file and add your API key
except KeyError:
    print("Error: ANTHROPIC_API_KEY not found in environment variables.")
    # Exit the script or handle the error as appropriate

LLM = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

search = DuckDuckGoSearchRun()

tool = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to reaseach for info"
    )
]

# Team Members

reasarcher = Agent(
    role="Tech Research",
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You Have a knack for dissecting complex data and presenting
    actionable insights""",
    verbose=True,
    allow_delegation=False,
    llm=LLM,
    tool=[tool]
)

writer = Agent(
    role='Tech Content Strategist',
    goal= 'Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for Your insightful and engaging article.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=False,
    tool=[tool]
)

# Tasks

task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify Key trends, breakthrough technologies, and potential industry impact.
    your final answer Must be a full analysis report""",
    agent=reasarcher
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Make it sound cool, avoid complex words so it doesn't sound like AI.
    Your final Answer Must be the full Blog post of at least 4 paragraphs.""",
    agent=writer
)

# Create CrewAI

Crew = Crew(
    agents=[reasarcher,writer],
    tasks=[task1,task2],
    verbose=2
)

# Start work!

# Main script execution
try:
    logger.info("Starting task execution")
    result = Crew.kickoff()
    logger.info("Task execution completed successfully")
    logger.info(f"Result: {result}")
except Exception as e:
    logger.error(f"An error occurred during task execution: {e}")
    # Handle the error or perform cleanup if necessary
