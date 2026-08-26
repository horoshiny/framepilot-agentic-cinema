from .tools import create_shot_plan, critique_shot

try:
    from google.adk.agents import Agent

    root_agent = Agent(
        name="cinematic_director",
        model="gemini-2.5-flash",
        description="An explainable previsualisation director for screenplay and storyboard workflows.",
        instruction=(
            "Translate narrative intent into controllable shot parameters. Call create_shot_plan first. "
            "Then call critique_shot on the resulting JSON. Explain only concise directing rationales; "
            "never reveal private chain-of-thought. Prefer restrained cinematic motion over decorative effects."
        ),
        tools=[create_shot_plan, critique_shot],
    )
except ImportError:
    root_agent = None

