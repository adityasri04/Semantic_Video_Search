import time
import os
import glob
import requests
from tkinter import filedialog
import tkinter as tk
import logging

log = logging.getLogger(__name__)

API_URL = "http://localhost:8000/api/v1"
USER_ID = "default_user"


def run_search(query):
    """
    Step 3: Perform search via FastAPI backend.
    """
    print(f"Searching for: '{query}'")
    t1 = time.perf_counter()

    try:
        response = requests.post(
            f"{API_URL}/search",
            data={"query": query, "user_id": USER_ID, "limit": 5}
        )
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        print(f"Search failed: {e}")
        return

    t2 = time.perf_counter()
    print(f"Search completed in {t2-t1:.3f}s\n")
    print(f"--- Search Results --- (query: '{query}')")
    if not results:
        print("No results found.")
    for idx, res in enumerate(results):
        print(
            f"{idx+1}. [{res.get('filename', 'N/A')}] "
            f"Time: {res.get('timestamp', 0):.2f}s | "
            f"Frame: {res.get('frame_idx', 'N/A')} | "
            f"Score: {res.get('score', 0):.4f} | "
            f"Video ID: {res.get('video_id', 'N/A')}"
        )
    print("----------------------\n")


def _poll_tasks(task_map: dict) -> None:
    """
    Poll multiple task IDs concurrently until all reach a terminal state.
    task_map: {task_id: filename}
    """
    pending = dict(task_map)  # copy

    while pending:
        time.sleep(2)
        done = set()
        for task_id, filename in list(pending.items()):
            try:
                r = requests.get(f"{API_URL}/status/{task_id}", timeout=5)
                if r.status_code == 200:
                    info   = r.json()
                    status = info.get("status", "unknown")
                    if status == "completed":
                        print(f"  ✓ [{filename}] Done!")
                        done.add(task_id)
                    elif status == "failed":
                        err = info.get("error", "unknown error")
                        print(f"  ✗ [{filename}] Failed: {err}")
                        done.add(task_id)
                    else:
                        print(f"  ⟳ [{filename}] {status}…")
                else:
                    print(f"  ? [{filename}] HTTP {r.status_code}")
            except Exception as e:
                print(f"  Status check error for {task_id}: {e}")
                done.add(task_id)

        for tid in done:
            del pending[tid]

    print("\nAll tasks finished.\n")


def add_videos_flow():
    """Upload one or more video files to the backend in a single request."""
    root = tk.Tk()
    root.withdraw()

    choice = input("Add a whole FOLDER? (y/n)\n : ")

    video_files = []
    if choice.lower() == "y":
        folder = filedialog.askdirectory(title="Select Video Folder")
        if not folder:
            return
        video_files = (
            glob.glob(os.path.join(folder, "*.mp4")) +
            glob.glob(os.path.join(folder, "*.mkv")) +
            glob.glob(os.path.join(folder, "*.mov")) +
            glob.glob(os.path.join(folder, "*.avi"))
        )
        if not video_files:
            print("No video files found in that folder.")
            return
    elif choice.lower() == "n":
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi")],
        )
        if not files:
            return
        video_files = list(files)
    else:
        print("Invalid input.")
        return

    print(f"\nSelected {len(video_files)} video(s):")
    for vf in video_files:
        print(f"  {os.path.basename(vf)}")

    # ── Send all files in ONE request ─────────────────────────────────
    print(f"\nUploading {len(video_files)} file(s) to {API_URL}/upload …")
    t_start = time.perf_counter()

    file_handles = []
    try:
        # Multipart: multiple "files" entries — FastAPI receives List[UploadFile]
        multipart = []
        for vf in video_files:
            fh = open(vf, "rb")
            file_handles.append(fh)
            multipart.append(
                ("files", (os.path.basename(vf), fh, "video/mp4"))
            )

        response = requests.post(
            f"{API_URL}/upload",
            data={"user_id": USER_ID},
            files=multipart,
        )
        response.raise_for_status()
        result = response.json()

    except Exception as e:
        print(f"Upload request failed: {e}")
        return
    finally:
        for fh in file_handles:
            fh.close()

    tasks = result.get("tasks", [])
    print(f"  {len(tasks)} task(s) queued. Polling for completion…\n")

    # ── Poll all tasks in parallel ────────────────────────────────────
    task_map = {t["task_id"]: t.get("filename", t["task_id"]) for t in tasks}
    _poll_tasks(task_map)

    elapsed = time.perf_counter() - t_start
    print(f"Total wall-clock time: {elapsed:.1f}s\n")


def remove_videos_flow():
    """Fetch the user's video list and interactively delete one."""
    print(f"\nFetching your video library (user: {USER_ID})…")
    try:
        r = requests.get(f"{API_URL}/videos", params={"user_id": USER_ID}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Could not fetch video list: {e}")
        return

    videos = data.get("videos", [])
    if not videos:
        print("No videos found in your library. Upload some first.")
        return

    print(f"\n--- Your Videos ({len(videos)}) ---")
    for i, v in enumerate(videos, 1):
        print(
            f"  {i:2}. [{v.get('filename', 'unknown')}]  "
            f"{v.get('frame_count', 0)} frames  |  ID: {v.get('video_id', 'N/A')}"
        )
    print("-----------------------------------")

    choice = input("\nEnter number to delete (or 'q' to cancel): ").strip()
    if choice.lower() == "q":
        return

    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(videos)):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input — enter a number.")
        return

    target = videos[idx]
    confirm = input(
        f"\nDelete '{target.get('filename')}' "
        f"({target.get('frame_count')} frames)? This cannot be undone. (y/n): "
    ).strip().lower()

    if confirm != "y":
        print("Cancelled.")
        return

    try:
        r = requests.delete(
            f"{API_URL}/video/{target['video_id']}",
            params={"user_id": USER_ID},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
        print(
            f"\n✓ Deleted '{target.get('filename')}' — "
            f"{result.get('frames_removed', '?')} frames removed from index."
        )
    except Exception as e:
        print(f"Delete failed: {e}")



def main():
    print("AI Video Search Engine Client Started.")
    print(f"Ensure the API is running at {API_URL}")

    while True:
        choice = input("\n1. Search\n2. Add Videos\n3. Remove Videos\n4. Exit\n : ")
        try:
            choice = int(choice)
        except (ValueError, TypeError):
            continue

        if choice == 1:
            query = input("Enter search query : ").strip()
            if query:
                run_search(query)
                input("\nPress Enter to continue…")

        elif choice == 2:
            add_videos_flow()

        elif choice == 3:
            remove_videos_flow()

        elif choice == 4:
            break

    print("Exited\n")


if __name__ == "__main__":
    main()
