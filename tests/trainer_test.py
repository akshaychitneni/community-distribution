#!/usr/bin/env python3

"""Trainer integration test using the Kubeflow SDK.

Follows the same pattern as pipeline_v2_test.py: use the SDK client
to create a training job, then poll until completion.

When the OptimizationJob CRD is installed (Kubeflow Trainer v2.4 onwards),
the test also runs a hyperparameter search. The Kubeflow SDK does not support
OptimizationJob in a released version yet, so that part uses the Kubernetes client.
"""

import json
import math
import sys
import time

from kubernetes import client as kubernetes_client
from kubernetes import config as kubernetes_config
from kubernetes.client.rest import ApiException

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer import CustomTrainer, TrainerClient

TRAINER_GROUP = "trainer.kubeflow.org"
TRAINER_VERSION = "v1alpha1"
OPTIMIZATION_JOB_CRD = "optimizationjobs.trainer.kubeflow.org"
OPTIMIZATION_JOB_LABEL = "trainer.kubeflow.org/optimization-job"
OPTIMIZATION_JOB_TIMEOUT_SECONDS = 600

# Each trial fits y = 2x + 1 with gradient descent, using the learning rate and
# epoch count that the controller injects, and reports the validation loss
# through the Trainer status server before it exits. It needs only the Python
# standard library. Adapted from the upstream example in kubeflow/trainer#4268.
OPTIMIZATION_TRIAL_SCRIPT = """
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

learning_rate = float(os.environ["KUBEFLOW_TRAINER_OPT_learning_rate"])
epochs = int(os.environ["KUBEFLOW_TRAINER_OPT_epochs"])

training_inputs = [-1.0, -0.5, 0.0, 0.5, 1.0]
weight, bias = 0.0, 0.0
for _ in range(epochs):
    errors = [weight * x + bias - (2 * x + 1) for x in training_inputs]
    weight -= learning_rate * 2 * sum(e * x for e, x in zip(errors, training_inputs)) / len(training_inputs)
    bias -= learning_rate * 2 * sum(errors) / len(training_inputs)

validation_inputs = [-0.75, -0.25, 0.25, 0.75]
loss = sum((weight * x + bias - (2 * x + 1)) ** 2 for x in validation_inputs) / len(validation_inputs)
print(f"learning_rate={learning_rate} epochs={epochs} validation_loss={loss}", flush=True)

payload = json.dumps({"trainerStatus": {
    "progressPercentage": 100,
    "lastUpdatedTime": datetime.now(timezone.utc).isoformat(),
    "metrics": [{"name": "validation_loss", "value": str(loss)}],
}}).encode()
tls_context = ssl.create_default_context(cafile=os.environ["KUBEFLOW_TRAINER_SERVER_CA_CERT"])
for attempt in range(3):
    token = Path(os.environ["KUBEFLOW_TRAINER_SERVER_TOKEN"]).read_text().strip()
    request = urllib.request.Request(
        os.environ["KUBEFLOW_TRAINER_SERVER_URL"],
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, context=tls_context, timeout=10) as response:
            response.read()
        break
    except urllib.error.HTTPError as error:
        if (error.code < 500 and error.code != 429) or attempt == 2:
            raise
    except (urllib.error.URLError, TimeoutError):
        if attempt == 2:
            raise
    time.sleep(5)
"""


def training_function():
    """Minimal training function with synthetic data.

    Uses a simple feedforward network on random tensors to verify
    that the Trainer SDK can create and run a TrainJob end-to-end.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    training_dataset = TensorDataset(torch.randn(64, 784), torch.randint(0, 10, (64,)))
    for inputs, targets in DataLoader(training_dataset, batch_size=32):
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss={loss.item():.4f}")

    print("Training complete")


def print_diagnostic_information(job_name, namespace):
    """Print diagnostic information for debugging timeout failures."""
    custom_objects_api = kubernetes_client.CustomObjectsApi()
    core_api = kubernetes_client.CoreV1Api()

    print("=== TrainJob status ===")
    try:
        trainjob = custom_objects_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            name=job_name,
        )
        print(json.dumps(trainjob, indent=2, default=str))
    except Exception as error:
        print(f"Failed to get TrainJob: {error}")

    print("=== Pods ===")
    try:
        pod_list = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"jobset.sigs.k8s.io/jobset-name={job_name}",
        )
        for pod in pod_list.items:
            print(f"Pod: {pod.metadata.name}  Phase: {pod.status.phase}")
            try:
                pod_logs = core_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=namespace,
                    tail_lines=50,
                )
                print(pod_logs)
            except Exception as log_error:
                print(f"Failed to get logs for {pod.metadata.name}: {log_error}")
    except Exception as error:
        print(f"Failed to list pods: {error}")

    print("=== Events ===")
    try:
        event_list = core_api.list_namespaced_event(namespace=namespace)
        sorted_events = sorted(
            event_list.items,
            key=lambda event: event.metadata.creation_timestamp or "",
        )
        for event in sorted_events[-20:]:
            print(
                f"{event.metadata.creation_timestamp}  "
                f"{event.reason}: {event.message}"
            )
    except Exception as error:
        print(f"Failed to list events: {error}")

    sys.stdout.flush()


def optimization_job_crd_installed():
    """Return whether the OptimizationJob CRD exists (Kubeflow Trainer v2.4 onwards)."""
    try:
        kubernetes_client.ApiextensionsV1Api().read_custom_resource_definition(
            OPTIMIZATION_JOB_CRD
        )
        return True
    except ApiException as error:
        if error.status == 404:
            return False
        raise


def build_optimization_job():
    """Build a random search over learning rate and epochs that minimizes validation loss."""
    return {
        "apiVersion": f"{TRAINER_GROUP}/{TRAINER_VERSION}",
        "kind": "OptimizationJob",
        "metadata": {"generateName": "trainer-optimizationjob-"},
        "spec": {
            "numTrials": 2,
            "parallelTrials": 2,
            "searchAlgorithm": {"random": {"seed": 42}},
            "objectives": [{"metric": "validation_loss", "direction": "Minimize"}],
            "parameters": [
                {
                    "name": "learning_rate",
                    "searchSpace": {
                        "logUniform": {"min": "0.01", "max": "0.2", "type": "Float"}
                    },
                },
                {
                    "name": "epochs",
                    "searchSpace": {"categorical": {"choices": ["10", "20", "40"]}},
                },
            ],
            "trainJobTemplate": {
                "spec": {
                    "runtimeRef": {
                        "apiGroup": TRAINER_GROUP,
                        "kind": "ClusterTrainingRuntime",
                        "name": "torch-distributed",
                    },
                    "trainer": {
                        "numNodes": 1,
                        "numProcPerNode": 1,
                        "resourcesPerNode": {
                            "requests": {"cpu": "1", "memory": "128Mi"},
                            "limits": {"cpu": "1", "memory": "128Mi"},
                        },
                        "image": "python:3.11-slim-bookworm",
                        "command": ["python3", "-c", OPTIMIZATION_TRIAL_SCRIPT],
                    },
                }
            },
        },
    }


def print_optimization_job_diagnostic_information(job_name, namespace):
    """Print the OptimizationJob, its trials and their pods for debugging failures."""
    custom_objects_api = kubernetes_client.CustomObjectsApi()
    core_api = kubernetes_client.CoreV1Api()

    print("=== OptimizationJob status ===")
    try:
        optimization_job = custom_objects_api.get_namespaced_custom_object(
            TRAINER_GROUP, TRAINER_VERSION, namespace, "optimizationjobs", job_name
        )
        print(json.dumps(optimization_job.get("status"), indent=2, default=str))
    except Exception as error:
        print(f"Failed to get OptimizationJob: {error}")

    print("=== Search algorithm pods ===")
    try:
        pod_list = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"{OPTIMIZATION_JOB_LABEL}={job_name}",
        )
        for pod in pod_list.items:
            print(f"Pod: {pod.metadata.name}  Phase: {pod.status.phase}")
    except Exception as error:
        print(f"Failed to list search algorithm pods: {error}")

    try:
        trials = custom_objects_api.list_namespaced_custom_object(
            TRAINER_GROUP,
            TRAINER_VERSION,
            namespace,
            "trainjobs",
            label_selector=f"{OPTIMIZATION_JOB_LABEL}={job_name}",
        )["items"]
    except Exception as error:
        print(f"Failed to list trial TrainJobs: {error}")
        trials = []
    for trial in trials:
        print_diagnostic_information(trial["metadata"]["name"], namespace)


def wait_for_optimization_job(job_name, namespace):
    """Wait until the OptimizationJob is Complete and fail early if it is Failed."""
    custom_objects_api = kubernetes_client.CustomObjectsApi()
    deadline = time.monotonic() + OPTIMIZATION_JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        optimization_job = custom_objects_api.get_namespaced_custom_object(
            TRAINER_GROUP, TRAINER_VERSION, namespace, "optimizationjobs", job_name
        )
        conditions = (optimization_job.get("status") or {}).get("conditions") or []
        for condition in conditions:
            if condition["status"] != "True":
                continue
            if condition["type"] == "Complete":
                return optimization_job
            if condition["type"] == "Failed":
                raise RuntimeError(
                    f"OptimizationJob {job_name} failed: "
                    f"{condition.get('reason')}: {condition.get('message')}"
                )
        time.sleep(5)
    raise TimeoutError(
        f"OptimizationJob {job_name} did not complete in "
        f"{OPTIMIZATION_JOB_TIMEOUT_SECONDS}s"
    )


def run_optimization_job_test(namespace):
    """Run a hyperparameter search and verify that the best trial is recorded."""
    custom_objects_api = kubernetes_client.CustomObjectsApi()
    optimization_job = build_optimization_job()
    job_name = custom_objects_api.create_namespaced_custom_object(
        TRAINER_GROUP,
        TRAINER_VERSION,
        namespace,
        "optimizationjobs",
        optimization_job,
    )["metadata"]["name"]
    print(f"Created OptimizationJob: {job_name}")

    try:
        try:
            completed_job = wait_for_optimization_job(job_name, namespace)
        except (RuntimeError, TimeoutError) as error:
            print(f"\n=== ERROR: {error} ===")
            print_optimization_job_diagnostic_information(job_name, namespace)
            sys.stdout.flush()
            raise

        trials = custom_objects_api.list_namespaced_custom_object(
            TRAINER_GROUP,
            TRAINER_VERSION,
            namespace,
            "trainjobs",
            label_selector=f"{OPTIMIZATION_JOB_LABEL}={job_name}",
        )["items"]
        expected_trials = optimization_job["spec"]["numTrials"]
        assert (
            len(trials) == expected_trials
        ), f"Expected {expected_trials} trial TrainJobs, found {len(trials)}"

        result = completed_job["status"]["result"]
        best_trial_name = result["trainJobName"]
        parameter_names = {parameter["name"] for parameter in result["parameters"]}
        assert parameter_names == {
            "learning_rate",
            "epochs",
        }, f"Unexpected parameters in the best result: {result['parameters']}"

        best_trial = custom_objects_api.get_namespaced_custom_object(
            TRAINER_GROUP, TRAINER_VERSION, namespace, "trainjobs", best_trial_name
        )
        metrics = best_trial["status"]["trainerStatus"]["metrics"]
        validation_loss = next(
            float(metric["value"])
            for metric in metrics
            if metric["name"] == "validation_loss"
        )
        assert math.isfinite(
            validation_loss
        ), f"Best trial reported a non-finite validation_loss: {validation_loss}"
        print(
            f"OptimizationJob completed: best trial {best_trial_name} with "
            f"{result['parameters']} and validation_loss={validation_loss}"
        )
    finally:
        custom_objects_api.delete_namespaced_custom_object(
            TRAINER_GROUP,
            TRAINER_VERSION,
            namespace,
            "optimizationjobs",
            job_name,
            body=kubernetes_client.V1DeleteOptions(propagation_policy="Foreground"),
        )


if __name__ == "__main__":
    namespace = sys.argv[1] if len(sys.argv) > 1 else "kubeflow-user-example-com"

    kubernetes_config.load_kube_config()

    trainer_client = TrainerClient(
        backend_config=KubernetesBackendConfig(namespace=namespace)
    )

    job_name = trainer_client.train(
        trainer=CustomTrainer(
            func=training_function,
            num_nodes=1,
            resources_per_node={"memory": "2Gi", "cpu": "1"},
            packages_to_install=["torch<3"],
            pip_index_urls=["https://download.pytorch.org/whl/cpu"],
        ),
    )
    print(f"Created TrainJob: {job_name}")

    try:
        trainer_client.wait_for_job_status(job_name, timeout=300, polling_interval=5)
        print("TrainJob completed successfully")
    except RuntimeError:
        print(f"\n=== ERROR: TrainJob {job_name} failed ===")
        print_diagnostic_information(job_name, namespace)
        sys.stdout.flush()
        raise
    except TimeoutError:
        print(f"\n=== TIMEOUT: TrainJob {job_name} did not complete in 300s ===")
        print_diagnostic_information(job_name, namespace)
        sys.stdout.flush()
        raise

    if optimization_job_crd_installed():
        run_optimization_job_test(namespace)
    else:
        print(
            f"{OPTIMIZATION_JOB_CRD} is not installed (Trainer earlier than v2.4); "
            "skipping the OptimizationJob test"
        )
