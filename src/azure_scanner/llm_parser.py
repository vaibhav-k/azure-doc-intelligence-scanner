from __future__ import annotations

import os
import sys

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from src.azure_scanner.models import MedicalPrescription


def parse_markdown_with_gpt(markdown_content: str) -> MedicalPrescription:
    """
    Parses Azure Document Intelligence Markdown output using Azure OpenAI GPT-4o
    with Structured Outputs conforming to the MedicalPrescription Pydantic model.

    Args:
        markdown_content: The Markdown content extracted from Azure Document Intelligence.

    Raises:
        ValueError: If the Azure OpenAI endpoint is not set or if the model fails to parse the response.

    Returns:
        MedicalPrescription: The parsed medical prescription data.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT must be set in environment variables.")

    try:
        # Support API Key auth if provided, otherwise fallback to Azure AD credentials
        if api_key:
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )

        completion = client.beta.chat.completions.parse(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert clinical data extraction assistant. "
                        "Accurately extract structured medical prescription and patient information "
                        "from the provided Document Intelligence Markdown text. Handle abbreviations, "
                        "handwriting notes, and incomplete fields gracefully."
                    ),
                },
                {"role": "user", "content": markdown_content},
            ],
            response_format=MedicalPrescription,
        )

        parsed_data = completion.choices[0].message.parsed
        if parsed_data is None:
            raise ValueError(
                "Model failed to parse response into MedicalPrescription schema."
            )
        return parsed_data

    except Exception as e:
        print(f"Error during Azure OpenAI markdown parsing: {e}", file=sys.stderr)
        raise
