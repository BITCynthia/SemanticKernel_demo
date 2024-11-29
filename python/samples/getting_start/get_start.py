from semantic_kernel import __version__

__version__

# Make sure paths are correct for the imports

import os
import sys

notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.append(grandparent_dir)


from dotenv import load_dotenv
import os

load_dotenv()

GLOBAL_LLM_SERVICE = os.getenv("GLOBAL_LLM_SERVICE")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPRNAI_API_VERSION = os.getenv('AZURE_OPRNAI_API_VERSION')

print(f"GLOBAL_LLM_SERVICE: {GLOBAL_LLM_SERVICE}")
print(f"AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: {AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}")
print(f"AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")
print(f"AZURE_OPENAI_API_KEY: {AZURE_OPENAI_API_KEY}")
print(f"AZURE_OPRNAI_API_VERSION: {AZURE_OPRNAI_API_VERSION}")

from semantic_kernel import Kernel

kernel = Kernel()
kernel

from services import Service

from samples.service_settings import ServiceSettings

service_settings = ServiceSettings.create(env_file_path=".env")

# Select a service to use for this notebook (available services: OpenAI, AzureOpenAI, HuggingFace)
selectedService = (
    Service.AzureOpenAI
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
print(f"Using service type: {selectedService}")
service_settings.model_config


# Remove all services so that this cell can be re-run without restarting the kernel
kernel.remove_all_services()

service_id = None
if selectedService == Service.OpenAI:
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    service_id = "default"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
        ),
    )
elif selectedService == Service.AzureOpenAI:
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            # api_key=AZURE_OPENAI_API_KEY,
            # endpoint=AZURE_OPENAI_ENDPOINT,
            # deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            # api_version=AZURE_OPRNAI_API_VERSION,
            env_file_path=".env",
            env_file_encoding=service_settings.env_file_encoding,
        ),
    )

plugin = kernel.add_plugin(parent_directory="../../../prompt_template_samples/", plugin_name="FunPlugin")

from semantic_kernel.functions import KernelArguments


import asyncio

async def main():
    joke_function = plugin["Joke"]

    joke = await kernel.invoke(
        joke_function,
        KernelArguments(input="time travel to dinosaur age", style="super silly"),
    )

    print(joke)

asyncio.run(main())