import torch
from typing import Union, List
from transformers import AutoTokenizer
import re
import numpy as np
import requests


def _value_inference_fastchat(
    model_name: str,
    input_str: Union[List[str], str],
    controller_addr="http://0.0.0.0:28777",
):
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Value Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {"input_str": input_str}
    response = requests.post(
        worker_addr + "/worker_value_inference",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    if response.status_code == 200:
        try:
            results = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"JSON decode error: Response content: {response.content}")
            # Handle error as needed, e.g., return a default value or raise an error
    else:
        print(f"Error: Received status code {response.status_code} with content: {response.content}")
        # Handle non-200 status here
    value = results["value"]
    return value
