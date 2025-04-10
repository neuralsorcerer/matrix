# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import ArgumentParser
from typing import Any, Dict, List

from fastapi import FastAPI
from google import genai
from ray import serve
from starlette.requests import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
@serve.ingress(app)
class GeminiDeployment:
    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def _transform_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Transform the given input messages to the format recognized by Google Gemini API.
        The Gemini API expects messages with 'role' and 'parts' keys, where 'parts' is a list of dictionaries containing the 'text' key.
        This function maps the input roles to either 'user' or 'model', as these are the only roles recognized by the API.
        Args:
            messages (List[Dict[str, str]]): A list of dictionaries containing the input messages with 'role' and 'content' keys.
        Returns:
            List[Dict[str, List[Dict[str, Any]]]]: A list of dictionaries containing the transformed messages in the Gemini API format.
        """
        role_mapping: Dict[str, str] = {
            "developer": "user",
            "system": "user",
            "assistant": "model",
        }
        transformed_contents: List[Dict[str, Any]] = [
            {
                "role": role_mapping.get(message["role"], message["role"]),
                "parts": [{"text": message["content"]}],
            }
            for message in messages
        ]
        return transformed_contents

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.debug(f"Request: {request}")
        completion_request = request.model_dump(exclude_unset=True)

        # gemini api expects only one message as input
        messages_transformed: List[Dict[str, Any]] = self._transform_message(
            completion_request.get("messages", [])
        )

        request_params = {
            "contents": messages_transformed,
            "config": {
                "temperature": completion_request.get("temperature", 0.6),
                "top_p": completion_request.get("top_p", 0.9),
                "seed": completion_request.get("seed", 42),
                "max_output_tokens": completion_request.get("max_tokens", 1024),
                "response_logprobs": completion_request.get("logprobs", False),
                "candidate_count": completion_request.get("n", 1),
            },
        }
        response = await self.client.aio.models.generate_content(
            model=self.model_name, **request_params
        )

        completion_response: Dict[str, Any] = {
            "id": response.response_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": [
                        candidate.finish_reason.value
                        for candidate in response.candidates
                    ],
                    "message": {"content": response.text, "role": "assistant"},
                }
            ],
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            },
        }

        return completion_response


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    argparse = ArgumentParser()
    argparse.add_argument("--api_key", type=str, required=True)
    argparse.add_argument("--model_name", type=str, required=True)

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(args=arg_strings)

    logging.log(logging.INFO, f"args: {args}")

    return GeminiDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.api_key,
        args.model_name,
    )
