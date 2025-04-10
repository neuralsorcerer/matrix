# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Union

import portalocker
import ray
import yaml
from ray.serve import scripts
from ray.serve._private.common import DeploymentID
from ray.serve.context import _get_global_client
from ray.serve.schema import ApplicationStatusOverview, ServeStatus
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from matrix.app_server.deploy_utils import (
    delete_apps,
    get_yaml_for_deployment,
    is_sglang_app,
    write_yaml_file,
)
from matrix.common.cluster_info import ClusterInfo
from matrix.utils.http import fetch_url
from matrix.utils.json import convert_to_json_compatible
from matrix.utils.ray import (
    ACTOR_NAME_SPACE,
    Action,
    get_ray_address,
    get_ray_dashboard_address,
    kill_matrix_actors,
)

logger = logging.getLogger("ray.serve")


class EndpointCache:
    def __init__(
        self, ray_address, deployment_id, endpoint_template, ttl=360, serve_app=True
    ):
        self.lock = asyncio.Lock()
        self.timestamp = None
        self.ttl = ttl
        self.serve_app = serve_app
        self.deployment_id = deployment_id
        self.endpoint_template = endpoint_template
        self.ray_address = ray_address
        if not serve_app:
            if not ray.is_initialized():
                ray.init(
                    address=ray_address,
                    ignore_reinit_error=True,
                )

            self.router_actor = ray.get_actor(
                f"{self.deployment_id.app_name}_router", ACTOR_NAME_SPACE
            )
        else:
            self.ray_address = ray_address.replace("ray:", "http:")
        self.ips = set()

    async def __call__(self, force_update=False):
        if time.time() - (self.timestamp or 0) > self.ttl or force_update:
            async with self.lock:
                if time.time() - (self.timestamp or 0) > self.ttl or force_update:
                    try:
                        if self.serve_app:
                            self.timestamp = time.time()
                            status, content = await fetch_url(
                                self.ray_address + "/api/serve/applications/",
                                headers={"Accept": "application/json"},
                            )
                            if status is not None and status == 200:
                                ray_query_result = json.loads(content)
                                head_ip = ray_query_result["controller_info"]["node_ip"]
                                self.ips = set([y["node_ip"] for x, y in ray_query_result["proxies"].items() if y["status"] == "HEALTHY" and y["node_ip"] != head_ip])  # type: ignore[attr-defined]
                            else:
                                raise Exception(f"status: {status}, {content}")
                        else:
                            self.ips = ray.get(
                                self.router_actor.get_running_replicas.remote()
                            )

                    except Exception as e:
                        logger.warning(f"Error fetching endpoints: {e}")

        return [self.endpoint_template.format(host=ip) for ip in self.ips if ip]


DEPLOYMENT_YAML = "deployment.yaml"
DEPLOYMENT_SGLANG_YAML = "deployment_sglang.yaml"


def get_head_http_host(cluster_info: ClusterInfo) -> str:
    assert cluster_info.hostname
    devvm = cluster_info.hostname.startswith("dev")
    return "localhost" if devvm else cluster_info.hostname


def deploy(
    cluster_dir: Path,
    cluster_info: ClusterInfo,
    action: str | Action = Action.REPLACE,
    applications: Optional[List[Dict[str, Union[str, int]]]] = None,
    yaml_config: Optional[str] = None,
    block: bool = False,
):
    """
    Deploy ray serve applications using either a yaml_config file or using the builtin template configured by applications.

    args:
    yaml_config: standard ray serve config file format.
    applications: array of dictionary, eg [{"model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct", "min_replica": 8, "max_replica": 8}]
        or '[{"app_type": "code", "pythonpath": "'"`pwd`/libs/codegen/xlformers/lib"'"}]'
    """

    yaml_filepath = str(cluster_dir / DEPLOYMENT_YAML)
    sglang_yaml_filepath = str(cluster_dir / DEPLOYMENT_SGLANG_YAML)

    if not ray.is_initialized():
        ray.init(
            address=get_ray_address(cluster_info),
            ignore_reinit_error=True,
            log_to_driver=False,
        )

    temp_dir = cluster_info.temp_dir
    assert temp_dir, "head temp_dir is None"
    os.environ["OUTLINES_CACHE_DIR"] = os.path.join(temp_dir, ".outlines")

    assert yaml_config is None or os.path.exists(
        yaml_config
    ), f"{yaml_config} not found"
    assert (applications is None) != (
        yaml_config is None
    ), "provide a yaml_config file or the applications"
    try:
        action = Action(action) if isinstance(action, str) else action
    except ValueError:
        raise ValueError(
            f"Invalid action '{action}', expected one of {[a.value for a in Action]}"
        )

    with portalocker.Lock(yaml_filepath, "a+", timeout=10) as yaml_file:
        with portalocker.Lock(
            sglang_yaml_filepath, "a+", timeout=10
        ) as sglang_yaml_file:
            yaml_file.seek(0)
            old = yaml.safe_load(yaml_file)
            if old is None:
                old_apps: List[Dict[str, Union[str, int]]] = []
            else:
                old_apps = old["applications"] or []
            sglang_yaml_file.seek(0)
            sglang_old = yaml.safe_load(sglang_yaml_file)
            if sglang_old is None:
                sglang_old_apps: List[Dict[str, Union[str, int]]] = []
            else:
                sglang_old_apps = sglang_old["applications"] or []
            existing_apps = old_apps + sglang_old_apps
            yaml_str = get_yaml_for_deployment(
                cluster_info, action, applications, yaml_config, existing_apps
            )
            update_apps = yaml.safe_load(yaml_str)

            if update_apps["applications"] is None:
                if action == Action.REPLACE:
                    # special case of remove everything
                    delete_apps(cluster_info, None)
                    write_yaml_file(yaml_file, sglang_yaml_file, update_apps)
            else:
                if action == Action.REMOVE:
                    delete_apps(cluster_info, update_apps["applications"])
                    remove_names = [app["name"] for app in update_apps["applications"]]
                    old_apps = [
                        app for app in old_apps if app["name"] not in remove_names
                    ]
                    sglang_old_apps = [
                        app
                        for app in sglang_old_apps
                        if app["name"] not in remove_names
                    ]
                    remaining = old or sglang_old
                    remaining["applications"] = old_apps + sglang_old_apps
                    write_yaml_file(yaml_file, sglang_yaml_file, remaining)
                else:
                    # separate deploy for serve and sglang
                    sglang_apps = [
                        app for app in update_apps["applications"] if is_sglang_app(app)
                    ]
                    if sglang_apps:
                        from matrix.app_server.llm import deploy_sglang_app

                        assert (
                            len(update_apps["applications"]) == 1
                        ), "only support 1 sglang app"
                        assert (
                            applications is not None and len(applications) == 1
                        ), "sglang does not support yaml deploy"
                        write_yaml_file(None, sglang_yaml_file, update_apps)
                        kill_matrix_actors(cluster_info)
                        deploy_sglang_app.deploy_app(
                            cluster_dir, cluster_info, applications[0]
                        )
                    else:
                        if action == Action.ADD:
                            # disjoint
                            old_app_names = [app["name"] for app in existing_apps]
                            new_app_names = [
                                app["name"] for app in update_apps["applications"]
                            ]
                            duplicates = set(old_app_names) & set(new_app_names)
                            assert not duplicates, f"Add to existing apps {duplicates}"

                            update_apps["applications"].extend(existing_apps)
                        serve_apps, _ = write_yaml_file(yaml_file, None, update_apps)
                        assert serve_apps["applications"]
                        scripts.deploy(
                            [
                                "--address",
                                get_ray_dashboard_address(cluster_info),
                                yaml_file.name,
                            ],
                            standalone_mode=False,
                        )
            return [app["name"] for app in (update_apps.get("applications") or [])]


def append_deploy(
    cluster_dir: Path,
    cluster_info: ClusterInfo,
    app: Dict[str, Union[str, int]],
) -> str:
    """
    this is used to deploy one application, appending to current deployments.
    for example, model checkpoint evaluation
    when done with this application, call deploy with Action.REMOVE
    """
    name = app.get("name")
    if name is None:
        hex_hash = hashlib.sha256(str(app.get("model_name")).encode()).digest()
        name = base64.b32encode(hex_hash).decode()[:8]
        app["name"] = name
    deploy(cluster_dir, cluster_info, Action.ADD, [app])
    return str(name)


def remove_temp_app(cluster_dir: Path, cluster_info: ClusterInfo, app_name: str):
    app = {"name": app_name}  # type: ignore[assignment]
    deploy(cluster_dir, cluster_info, Action.REMOVE, [app])  # type: ignore[list-item]
    return app


def status(cluster_info: ClusterInfo, replica):
    ray_dashboard_url = get_ray_dashboard_address(cluster_info)
    subprocess.run(["serve", "status", "--address", ray_dashboard_url])
    subprocess.run(
        [
            "ray",
            "list",
            "actors",
            "--address",
            ray_dashboard_url,
            "--filter",
            "ray_namespace=matrix",
            "--filter",
            "state!=DEAD",
            "--limit",
            "10000",
        ]
    )
    if replica:
        print("\n\nReplica: " + "-" * 8)
        os.environ["RAY_ADDRESS"] = get_ray_address(cluster_info)
        _client = _get_global_client()
        replicas = ray.get(_client._controller._all_running_replicas.remote())  # type: ignore[union-attr]
        json_compatible_replicas = convert_to_json_compatible(replicas)
        print(json.dumps(json_compatible_replicas, indent=2))


def get_app_metadata(
    cluster_dir: Path,
    cluster_info: ClusterInfo,
    app_name: str,
    endpoint_ttl_sec: int = 5,
    model_name: Optional[str] = None,
    head_only: bool = False,
) -> Dict[str, Any]:
    http_port, grpc_port = None, None

    def get_app(deployment):
        nonlocal http_port, grpc_port

        yaml_config = str(cluster_dir / deployment)
        if not os.path.exists(yaml_config):
            print(f"config does not exist {yaml_config}")
            return None
        with open(yaml_config, "r") as file:
            data = yaml.safe_load(file)
        if data is None:
            print(f"empty config {yaml_config}")
            return None

        http_port = data["http_options"]["port"]
        grpc_port = data["grpc_options"]["port"]
        app = [
            a
            for a in (data["applications"] or [])
            if (
                (app_name and a["name"] == app_name)
                or (model_name and a["args"]["model"] == model_name)
            )
        ]
        if len(app) == 1:
            return app[0]
        else:
            return None

    serve_app = True
    app = get_app(DEPLOYMENT_YAML)
    if app is None:
        print("Nothing found. try sglang deployment")
        serve_app = False
        app = get_app(DEPLOYMENT_SGLANG_YAML)

    assert app, f"uknown app_name {app_name} within deployment {app}"

    prefix = app["route_prefix"].strip("/")  # type: ignore
    model = app["args"].get("model")  # type: ignore
    deployment_name = app["deployments"][0]["name"]  # type: ignore
    use_grpc = "GrpcDeployment" in deployment_name
    if serve_app:
        if "code" in deployment_name.lower() or "hello" in deployment_name.lower():
            endpoint_template = f"http://{{host}}:{http_port}/{prefix}"
        else:
            endpoint_template = (
                f"http://{{host}}:{http_port}/{prefix}/v1"
                if not use_grpc
                else f"{{host}}:{grpc_port}"
            )
    else:
        endpoint_template = f"http://{{host}}:{cluster_info.sglang_http_port}/v1"
    metadata = {
        "name": app_name,
        "http_port": http_port,
        "grpc_port": grpc_port,
        "route_prefix": prefix,
        "model_name": model,
        "deployment_name": deployment_name,
        "use_grpc": use_grpc,
        "endpoint_template": endpoint_template,
    }

    head = metadata["endpoint_template"].format(host=get_head_http_host(cluster_info))
    if head_only:

        async def dummy_updater():
            return head

        endpoint_cache = dummy_updater
        workers = []
    else:
        deployment_id = DeploymentID(
            name=metadata["deployment_name"], app_name=metadata["name"]
        )
        endpoint_cache = EndpointCache(
            (
                get_ray_dashboard_address(cluster_info)
                if serve_app
                else get_ray_address(cluster_info)
            ),
            deployment_id,
            metadata["endpoint_template"],
            ttl=endpoint_ttl_sec,
            serve_app=serve_app,
        )
        workers = asyncio.run(endpoint_cache())
    metadata["endpoints"] = {
        "head": head,
        "workers": workers,
        "updater": endpoint_cache,
    }

    return metadata


def inference(
    cluster_dir: Path,
    cluster_info: ClusterInfo,
    app_name: str,
    output_jsonl: str,
    input_jsonls: str,
    load_balance: bool = True,
    **kwargs,
):
    from matrix.app_server.llm.query_llm import main as query

    metadata = get_app_metadata(cluster_dir, cluster_info, app_name)
    assert cluster_info.hostname
    local_mode = cluster_info.hostname.startswith("devvm")

    async def get_one_endpoint() -> str:
        if not load_balance:
            return metadata["endpoint_template"].format(
                host="localhost" if local_mode else cluster_info.hostname
            )
        else:
            ips = await metadata["endpoints"]["updater"]()
            assert ips
            host = random.choice(ips)
            return host

    return asyncio.run(
        query(
            get_one_endpoint,
            output_jsonl,
            input_jsonls,
            model=metadata["model_name"],
            app_name=metadata["name"],
            **kwargs,
        )
    )
