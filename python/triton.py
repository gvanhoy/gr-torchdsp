#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Peraton Labs, Inc..
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import tritonclient.http as client
import numpy as np

INPUT_TO_TYPE_MAP = {
    "INT16": "short",
    "INT32": "int",
    "FLOAT32": "float",
    "FP32": "float"
}


def get_model_input_name(triton_url: str, model_name: str):
    """
        {
            'name': 'simple', 
            'versions': ['1'], 
            'platform': 'python', 
            'inputs': [{'name': 'INPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'INPUT1', 'datatype': 'INT32', 'shape': [1, 16]}], 
            'outputs': [{'name': 'OUTPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'OUTPUT1', 'datatype': 'INT32', 'shape': [1, 16]}]
        }

    Args:
        triton_url (str): _description_
        model_name (str): _description_
    """

    _client = client.InferenceServerClient(triton_url)
    input_meta = _client.get_model_metadata(model_name)["inputs"][0]
    return input_meta["name"]


def get_model_input_shape(triton_url: str, model_name: str):
    """
        {
            'name': 'simple', 
            'versions': ['1'], 
            'platform': 'python', 
            'inputs': [{'name': 'INPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'INPUT1', 'datatype': 'INT32', 'shape': [1, 16]}], 
            'outputs': [{'name': 'OUTPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'OUTPUT1', 'datatype': 'INT32', 'shape': [1, 16]}]
        }

    Args:
        triton_url (str): _description_
        model_name (str): _description_
    """

    _client = client.InferenceServerClient(triton_url)
    input_meta = _client.get_model_metadata(model_name)["inputs"][0]
    return input_meta["shape"]


def get_model_input_size(triton_url: str, model_name: str):
    """
        {
            'name': 'simple', 
            'versions': ['1'], 
            'platform': 'python', 
            'inputs': [{'name': 'INPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'INPUT1', 'datatype': 'INT32', 'shape': [1, 16]}], 
            'outputs': [{'name': 'OUTPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'OUTPUT1', 'datatype': 'INT32', 'shape': [1, 16]}]
        }

    Args:
        triton_url (str): _description_
        model_name (str): _description_
    """

    _client = client.InferenceServerClient(triton_url)
    input_meta = _client.get_model_metadata(model_name)["inputs"][0]
    return np.prod(input_meta["shape"])


def get_model_input_data_type(triton_url: str, model_name: str):
    """
        {
            'name': 'simple', 
            'versions': ['1'], 
            'platform': 'python', 
            'inputs': [{'name': 'INPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'INPUT1', 'datatype': 'INT32', 'shape': [1, 16]}], 
            'outputs': [{'name': 'OUTPUT0', 'datatype': 'INT32', 'shape': [1, 16]}, {'name': 'OUTPUT1', 'datatype': 'INT32', 'shape': [1, 16]}]
        }

    Args:
        triton_url (str): _description_
        model_name (str): _description_
    """

    _client = client.InferenceServerClient(triton_url)
    input_meta = _client.get_model_metadata(model_name)["inputs"][0]
    return INPUT_TO_TYPE_MAP[input_meta["datatype"]]


def get_available_models(triton_url: str):
    """
        [
            {'name': 'add', 'version': '1', 'state': 'READY'}, 
            {'name': 'alexnet', 'version': '1', 'state': 'READY'}, 
            {'name': 'beamform', 'version': '1', 'state': 'READY'}, 
            {'name': 'efficientnet_b0', 'version': '1', 'state': 'READY'},
        ]

    Args:
        triton_url (str): _description_
        model_name (str): _description_
    """

    _client = client.InferenceServerClient(triton_url)
    models = _client.get_model_repository_index()
    print([model["name"] for model in models])
    return [model["name"] for model in models]


if __name__ == "__main__":
    get_available_models("localhost:8000")
