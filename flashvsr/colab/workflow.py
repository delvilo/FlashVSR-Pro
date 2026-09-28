"""Stage locally, checkpoint every completed segment, publish verified results."""

from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from flashvsr.config import validate_input
from flashvsr.jobs import file_identity
from flashvsr.long_video import run_long
from flashvsr.media import MediaTools
from flashvsr.observability import configure_logging, environment, write_report
from .storage import copy_verified, local_lock, owned_path, prepare_models, require_local_space

logger = logging.getLogger(__name__)


def project_identity(project):
    project = Path(project)
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=project, text=True)
    if status.strip():
        raise RuntimeError("Colab jobs require an unmodified Git checkout; commit changes or select a clean revision")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()


def validate_job_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Job name must be 1–80 letters, digits, underscores or hyphens")
    return name


class CheckpointMirror:
    def __init__(self, store, name, local_job, log_path):
        self.store = store
        self.remote = Path("jobs") / validate_job_name(name) / "checkpoint"
        self.local = Path(local_job)
        self.log = Path(log_path)

    def __call__(self, job):
        # run_long still holds its local JobRecord lock. Copy data before the
        # manifest, so even a killed VM cannot publish an unuploaded segment.
        for item in job.data["segments"]:
            if item["status"] == "done":
                relative = Path("segments") / f'{item["index"]:06d}.mkv'
                self.store.copy(job.segment_path(item["index"]), self.remote / relative, item["identity"])
        self.store.write_json(self.remote / "job.json", job.data)
        if self.log.is_file():
            self.store.copy(self.log, self.remote.parent / "run.jsonl")

    def restore(self):
        # A same-VM retry can include a newer segment whose Drive upload failed.
        # Preserve that local record; run_long rechecks each hash before reuse.
        if (self.local / "job.json").is_file():
            return
        record = self.store.path(self.remote / "job.json")
        if not record.is_file():
            return  # Interrupted before the first segment; start its local job.
        data = json.loads(record.read_text())
        for item in data.get("segments", []):
            if item["status"] != "done":
                continue
            relative = Path("segments") / f'{item["index"]:06d}.mkv'
            target = owned_path(self.local, relative)
            try:
                self.store.restore(self.remote / relative, target, item["identity"])
            except (FileNotFoundError, ValueError):
                item["status"] = "pending"
                logger.warning("Checkpoint %s is missing/corrupt; it will be recomputed", item["index"])
        write_report(self.local / "job.json", data)

    def cleanup(self):
        record = self.store.path(self.remote / "job.json")
        if record.is_file():
            data = json.loads(record.read_text())
            for item in data.get("segments", []):
                relative = Path("segments") / f'{item["index"]:06d}.mkv'
                for path in (self.store.path(self.remote / relative), owned_path(self.local, relative),
                             owned_path(self.local, f"{relative}.json")):
                    path.unlink(missing_ok=True)


def run_job(project, store, local_root, name, config, source=None, *, resume=False, segment_frames=129,
            keep_temp=False):
    """Drive run.json pins code/runtime/config; completion.json commits all results."""
    name = validate_job_name(name)
    local_root = Path(local_root).resolve()
    local = local_root / "jobs" / name
    remote = Path("jobs") / name
    local.mkdir(parents=True, exist_ok=True)
    log = local / "run.jsonl"
    configure_logging(log_file=log)
    config = replace(config, model_dir=local_root / "models/v1.1")
    requested = {"schema": 1, "job": name, "code_commit": project_identity(project),
                 "runtime": environment(), "parameters": config.as_dict(),
                 "segment_frames": segment_frames, "local_root": str(local_root)}
    with local_lock(local_root / ".colab.lock"):
        store.recover_partials()
        record_path = store.path(remote / "run.json")
        if resume:
            if not record_path.is_file():
                raise FileNotFoundError(f"No saved job: {name}")
            record = json.loads(record_path.read_text())
            if {key: record.get(key) for key in requested} != requested:
                raise ValueError("Resume settings/code/runtime changed. Select the saved Git commit and the same "
                                 "Python/packages/settings, or use a new job name.")
        else:
            if (record_path.exists() or (local / "checkpoint/job.json").exists()
                    or store.path(Path("results") / name).exists()):
                raise FileExistsError(f"Job {name} already exists; enable resume or choose another name")
            if source is None:
                raise ValueError("An input video is required for a new job")
            source = validate_input(source)
            if not source.is_file():
                raise ValueError("Colab jobs require a video file")
            original = file_identity(source)
            if source.is_relative_to(store.root):
                relative = source.relative_to(store.root)
                store.path(relative)  # Reject symlinks before retaining a reference.
            else:
                relative = remote / f"source{source.suffix.lower()}"
                # A reset during initial registration can leave a copied source
                # without run.json. Reuse only that exact orphan, never overwrite.
                existing = store.path(relative)
                if existing.exists() and file_identity(existing) != original:
                    raise FileExistsError(f"Job {name} contains another input; choose a new job name")
                if store.path(remote).exists() and any(path != existing for path in store.path(remote).iterdir()):
                    raise FileExistsError(f"Job {name} contains unrecognized files; choose a new job name")
                store.copy(source, relative, original)
            record = {**requested, "input": {"path": str(relative), "identity": original},
                      "input_name": source.name}
            store.write_json(remote / "run.json", record)
        if source is not None and resume and file_identity(validate_input(source)) != record["input"]["identity"]:
            raise ValueError("Resume input differs from the saved job")

        completed = store.path(remote / "completion.json")
        if completed.is_file():
            completion = json.loads(completed.read_text())
            if completion.get("job") != name:
                raise ValueError("Invalid completion record")
            for relative, identity in completion["files"].items():
                if file_identity(store.path(relative)) != identity:
                    raise ValueError(f"Saved result is missing/corrupt: {relative}; use a new job to regenerate")
            MediaTools().verify_video(store.path(completion["video"]))
            logger.info("Completed result verified; no model load: %s", store.path(completion["video"]))
            return completion

        prepare_models(store, local_root, config.mode)
        input_info = record["input"]
        local_source = local / f"input{Path(input_info['path']).suffix}"
        require_local_space(local, input_info["identity"]["bytes"] if not local_source.is_file() else 0)
        copy_verified(store.path(input_info["path"]), local_source, input_info["identity"])
        mirror = CheckpointMirror(store, name, local / "checkpoint", log)
        mirror.restore()
        destination = local / "output/FlashVSR_input_Final.mp4"
        report_path = Path(f"{destination}.json")
        result_root = Path("results") / name
        try:
            run_long(config, local_source, destination.parent, segment_frames=segment_frames,
                     work_dir=mirror.local, resume=(mirror.local / "job.json").is_file(),
                     keep_temp=True, checkpoint=mirror)
            MediaTools().verify_video(destination)
            files = {}
            for path, relative in ((destination, result_root / "enhanced.mp4"),
                                   (report_path, result_root / "report.json"),
                                   (log, result_root / "run.jsonl")):
                files[str(relative)] = store.copy(path, relative)
            files[str(result_root / "parameters.json")] = store.write_json(result_root / "parameters.json", record)
            setup_report = local_root / "setup.json"
            setup = json.loads(setup_report.read_text()) if setup_report.is_file() else {"runtime": environment()}
            files[str(result_root / "environment.json")] = store.write_json(result_root / "environment.json", setup)
            completion = {"schema": 1, "status": "ok", "job": name,
                          "video": str(result_root / "enhanced.mp4"), "files": files}
            # Commit after every result has been read back. A failed upload keeps
            # local output and all segment checkpoints for a later retry.
            store.write_json(remote / "completion.json", completion)
            if not keep_temp:
                try:
                    mirror.cleanup()
                except OSError:
                    logger.exception("Result saved; some temporary segments could not be removed")
            logger.info("Saved verified video and reports: %s", store.path(result_root))
            return completion
        except BaseException:
            logger.exception("Job stopped; resume %s to reuse saved segments", name)
            try:
                store.copy(log, remote / "run.jsonl")
                if report_path.is_file():
                    store.copy(report_path, remote / "last-report.json")
            except (OSError, ValueError):
                logger.exception("Could not save failure logs; local files remain at %s", local)
            raise


def list_jobs(store):
    directory = store.path("jobs")
    results = []
    if not directory.exists():
        return results
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or not (path / "run.json").is_file():
            continue
        try:
            record = json.loads(store.path(path.relative_to(store.root) / "run.json").read_text())
            saved = path / "checkpoint/job.json"
            checkpoint = json.loads(saved.read_text()) if saved.is_file() else {}
            results.append({"job": path.name, "completed": (path / "completion.json").is_file(),
                            "segments_saved": sum(i["status"] == "done" for i in checkpoint.get("segments", [])),
                            "code_commit": record["code_commit"], "parameters": record["parameters"],
                            "segment_frames": record["segment_frames"], "runtime": record["runtime"]})
        except (ValueError, KeyError, OSError):
            results.append({"job": path.name, "error": "Unreadable job metadata"})
    return results


def validate_gpu(project, store, local_root):
    """Run the existing GPU gate locally and persist its evidence within budget."""
    project, local_root = Path(project).resolve(), Path(local_root).resolve()
    with local_lock(local_root / ".colab.lock"):
        store.recover_partials()
        directory = local_root / "gpu-validation"
        directory.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
        remote = Path("diagnostics/gpu-validation") / run.name
        env = dict(os.environ, FLASHVSR_MODEL_PATH=str(local_root / "models/v1.1"))
        command = [sys.executable, str(project / "scripts/validate_gpu.py"), "--include-long",
                   "--output-dir", str(run)]
        passed = False
        try:
            with (run / "runner.log").open("w") as log:
                subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            passed = True
        finally:
            try:
                files = {str(path.relative_to(run)): store.copy(path, remote / path.relative_to(run))
                         for path in sorted(run.rglob("*")) if path.is_file()}
                store.write_json(remote / "saved.json", {"status": "passed" if passed else "failed", "files": files})
            except (OSError, ValueError):
                if passed:
                    raise
                logger.exception("Could not save GPU failure evidence; local files: %s", run)
        return {"status": "passed", "directory": str(store.path(remote))}
