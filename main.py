from crewai import Agent , Crew , Task
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_anthropic import ChatAnthropic
import os

Claude_3 = os.environ["ANTHROPIC_API_kEY"] = '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'

LLM = ChatAnthropic(temperature=0,model_name="claude-3-opus-2024229")

search = DuckDuckGoSearchRun()

tool = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to reaseach for info"
    )
]

#Team Member

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
    tools=[tool]
)


writer = Agent(
    role='Tech Content Strategist',
    goal= 'Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for Your insightful and engaging article.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=False,
    tools=[tool]
)


#Tasks

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



#create Crewai

Crew = Crew(
    agents=[reasarcher,writer],
    tasks=[task1,task2],
    verbose=2
)


#start work!

result = Crew.kickoff()
print(result)