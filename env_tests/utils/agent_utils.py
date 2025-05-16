import logging
import sys
logger = logging.getLogger(__name__)

from env_tests.agents import COGACTAgent, PI0Agent, GR00TAgent, LLMAgent


def create_vla_agent(args):
    """Create and initialize the appropriate VLA agent based on arguments."""
    vla = args.vla_agent.upper()
    vla_agent = getattr(sys.modules['env_tests.agents'], f"{vla}Agent")(  # Use getattr to dynamically create the class
        host=args.host,
        port=args.port,
        cameras=args.cameras
    )
    
    # Connect to the agent
    if vla_agent:
        success = vla_agent.connect()
        if not success:
            logger.error(f"Failed to connect to VLA agent")
            return None
    
    return vla_agent
    
def create_hierarchical_agent(args, vla_agent, env):
    """Create and initialize the hierarchical LLM agent."""
    logger.info("Creating Hierarchical VLA agent with LLM")
    
    # Create LLM agent
    llm_agent = LLMAgent(
        vla_agent=vla_agent,
        llm_provider=args.llm_provider,
        api_key=args.api_key,
        model=args.llm_model,
        observation_frequency=args.observation_frequency,
        env=env,
        cameras=args.cameras
    )
    
    return llm_agent