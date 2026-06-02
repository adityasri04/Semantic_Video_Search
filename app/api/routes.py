"""
app/api/routes.py
=================
FastAPI route handlers.

All model inference goes through SigLipEngine (ONNX Runtime) – no PyTorch
is needed at request time, keeping the dependency footprint small on CPU-only
deployments.
"""

import logging
import os
import shutil
import uuid
from typing import List

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile

from app.core.config import settings
from app.engine.models import SigLipEngine
from app.services.qdrant import qdrant_service
from app.services.tasks import process_video_background, tasks_db

log = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """
    Returns Qdrant connectivity status, mode, and model availability.
    Hit this first when debugging connection issues.
    """
    import os as _os
    qdrant_info = qdrant_service.health_check()

    model_path = os.path.join(
        settings.ONNX_MODEL_DIR, settings.ONNX_MODEL_FILE
    )
    model_ok = os.path.exists(model_path)

    return {
        "api": "ok",
        "qdrant": qdrant_info,
        "model": {
            "status": "ok" if model_ok else "missing",
            "path": model_path,
        },
    }


# ---------------------------------------------------------------------------
# Debug — Phase 3 observability
# ---------------------------------------------------------------------------

@router.get("/debug/collection")
async def debug_collection():
    """
    Quick diagnostic: shows how many vectors are stored in Qdrant.
    Use this after uploading a video to confirm embeddings were actually stored.
    Also runs a sample ONNX inference to confirm text-embedding dimension.
    """
    # ── Qdrant point count ────────────────────────────────────────────────────
    try:
        qdrant_service._ensure_collection()
        info = qdrant_service.client.get_collection(settings.QDRANT_COLLECTION)
        qdrant_info = {
            "collection": settings.QDRANT_COLLECTION,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as exc:
        qdrant_info = {"error": str(exc)}

    # ── ONNX model info ───────────────────────────────────────────────────────
    model_info = {"status": "not_checked"}
    try:
        siglip = SigLipEngine()
        import numpy as _np
        dummy_ids  = _np.zeros((1, 64), dtype=_np.int64)
        dummy_mask = _np.ones((1, 64),  dtype=_np.int64)
        emb = siglip.get_text_features(dummy_ids, dummy_mask)
        model_info = {
            "status": "ok",
            "mode": "split" if siglip.is_split_mode else "combined",
            "text_embedding_dim": int(emb.shape[-1]),
            "onnx_inputs":  sorted(siglip._input_names),
            "onnx_outputs": siglip._output_names,
        }
        if siglip.is_split_mode:
            model_info["vision_inputs"]  = sorted(siglip._vision_inputs)
            model_info["vision_outputs"] = siglip._vision_outputs
            model_info["text_inputs"]    = sorted(siglip._text_inputs)
            model_info["text_outputs"]   = siglip._text_outputs
    except Exception as exc:
        model_info = {"status": "error", "detail": str(exc)}

    return {"qdrant": qdrant_info, "model": model_info}


# ---------------------------------------------------------------------------
# Upload & ingestion
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_videos(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Accept one OR more video files and start a background processing task
    for each.  All tasks run concurrently — useful for batch ingestion.

    Returns a list of task objects so the client can poll each independently.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    for file in files:
        # Sanitise extension; default to mp4 if filename has no dot
        raw_name = file.filename or "upload"
        ext = raw_name.rsplit(".", 1)[-1] if "." in raw_name else "mp4"
        task_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_DIR, f"{task_id}.{ext}")

        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        tasks_db[task_id] = {"status": "queued", "filename": raw_name}
        background_tasks.add_task(process_video_background, task_id, temp_path, user_id)

        results.append({"task_id": task_id, "filename": raw_name, "status": "queued"})
        log.info("Queued task %s for file '%s' (user=%s)", task_id, raw_name, user_id)

    return {"tasks": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    status = tasks_db.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

@router.post("/search")
async def search_video(
    query: str = Form(...),
    user_id: str = Form(...),
    limit: int = 5,
):
    """
    Embed *query* with the SigLIP text encoder (ONNX) and search Qdrant.

    The text encoder path uses numpy inputs directly – no PyTorch needed.
    Returns a list of results sorted by relevance score (highest first).
    """
    log.info("Search request: query='%s', user_id='%s', limit=%d", query, user_id, limit)

    # ── 1. Load model (singleton – already in memory after first call) ──────
    try:
        siglip = SigLipEngine()
        _, processor, _ = siglip.get_components()
    except Exception as exc:
        log.error("Failed to load SigLipEngine: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Did you run tools/quantize_model.py? Error: {exc}",
        )

    # ── 2. Tokenise query ────────────────────────────────────────────────────
    try:
        inputs = processor(
            text=[query],
            return_tensors="np",         
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        input_ids = inputs["input_ids"].astype(np.int64)

        # attention_mask is required by SigLIP tokenizer
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.astype(np.int64)

        log.debug(
            "Tokenised query: input_ids shape=%s, attention_mask shape=%s",
            input_ids.shape,
            attention_mask.shape if attention_mask is not None else "None",
        )

    except Exception as exc:
        log.error("Tokenisation failed for query '%s': %s", query, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tokenisation failed: {exc}")

    # ── 3. Run ONNX text encoder ─────────────────────────────────────────────
    try:
        text_embedding = siglip.get_text_features(input_ids, attention_mask)
        # text_embedding shape: [1, 768]  →  take the first (and only) row
        text_vector = text_embedding[0].tolist()  # Python list of 768 floats

        log.debug(
            "Text embedding generated: dim=%d, first5=%s",
            len(text_vector),
            [f"{v:.4f}" for v in text_vector[:5]],
        )

    except Exception as exc:
        log.error("ONNX inference failed for query '%s': %s", query, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    # ── 4. Vector search in Qdrant ───────────────────────────────────────────
    try:
        results = qdrant_service.search(user_id, text_vector, limit)
    except Exception as exc:
        log.error("Qdrant search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Vector search failed: {exc}")

    if not results:
        log.info(
            "No results found for query='%s', user_id='%s'. "
            "Make sure videos have been uploaded and processed for this user.",
            query, user_id,
        )

    # ── 5. Format and return ───────────────────────────────────────────────────────
    return [
        {
            "score":     res.score,
            "timestamp": res.payload.get("timestamp"),
            "frame_idx": res.payload.get("frame_idx"),
            "video_id":  res.payload.get("video_id"),
            "filename":  res.payload.get("filename", ""),
        }
        for res in results
    ]


# ---------------------------------------------------------------------------
# Video management
# ---------------------------------------------------------------------------

@router.get("/videos")
async def list_videos(user_id: str = Query(..., description="User whose video library to list")):
    """
    List all videos indexed for *user_id*, with frame counts and video IDs.
    Use the returned video_id values with DELETE /video/{video_id} to remove videos.
    """
    try:
        videos = qdrant_service.list_user_videos(user_id)
    except Exception as exc:
        log.error("list_user_videos failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not list videos: {exc}")

    return {"user_id": user_id, "videos": videos, "count": len(videos)}


@router.delete("/video/{video_id}")
async def delete_video(
    video_id: str,
    user_id: str = Query(..., description="Owner of the video — only their data is touched"),
):
    """
    Delete all embedding vectors for *video_id* that belong to *user_id*.

    Safe to call on a video_id that no longer exists (returns frames_removed=0).
    Does NOT delete the original video file — only the search index entries.
    """
    try:
        result = qdrant_service.delete_video(user_id=user_id, video_id=video_id)
    except Exception as exc:
        log.error("delete_video failed for video_id='%s': %s", video_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Delete failed: {exc}")

    log.info("Deleted video_id='%s' for user='%s'.", video_id, user_id)
    return result


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------
# NOTE: These endpoints have no authentication in the prototype.
#       Before moving to production, add an API key or JWT middleware.
# ---------------------------------------------------------------------------

@router.get("/admin/users")
async def admin_list_users():
    """
    Admin: list every user_id in the collection with their video count and
    total indexed frames.  Requires a full collection scan — fast for prototype
    scale, consider a metadata sidecar for very large deployments.
    """
    try:
        users = qdrant_service.list_users()
    except Exception as exc:
        log.error("admin_list_users failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not list users: {exc}")

    return {"users": users, "total_users": len(users)}


@router.delete("/admin/user/{user_id}")
async def admin_delete_user(user_id: str):
    """
    Admin: permanently delete ALL data for *user_id* — every video, every
    embedding frame.  This operation is IRREVERSIBLE.

    Returns a summary: how many videos and frames were removed.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    try:
        result = qdrant_service.delete_user(user_id=user_id)
    except Exception as exc:
        log.error("admin_delete_user failed for user='%s': %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Delete failed: {exc}")

    return result
