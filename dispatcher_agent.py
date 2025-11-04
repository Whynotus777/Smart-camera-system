import json
from datetime import datetime
import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
LOG_FILE = "task_log.jsonl"


def connect_to_redis() -> redis.Redis:
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    client.ping()
    print(f"[Dispatcher] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return client


def get_best_robot(task_type: str) -> str:
    task_type = (task_type or "").lower()
    if task_type == "theft":
        return "SecurityBot_01"
    if task_type == "spill":
        return "CleanBot_01"
    return "SupportBot_01"


def determine_task_payload(task_type: str, event_data: dict) -> dict:
    task_type = (task_type or "").lower()
    if task_type == "theft":
        return {
            "task_name": "navigate_and_observe",
            "target_global_id": event_data.get("global_id"),
        }
    if task_type == "spill":
        return {
            "task_name": "dispatch_cleaning_bot",
            "target_zone": event_data.get("camera"),
        }
    return {
        "task_name": "monitor_event",
        "details": {"reason": event_data.get("reason"), "camera": event_data.get("camera")},
    }


def main():
    try:
        redis_client = connect_to_redis()
    except Exception as exc:
        print(f"[Dispatcher] Failed to connect to Redis: {exc}")
        return

    pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
    pubsub.psubscribe("store:events:*")
    print("[Dispatcher] Listening for events on pattern 'store:events:*'")

    for message in pubsub.listen():
        if message["type"] not in {"pmessage", "message"}:
            continue

        try:
            topic = message.get("channel") or message.get("pattern")
            topic = topic.decode("utf-8") if isinstance(topic, bytes) else str(topic)
            raw_data = message.get("data")
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")

            event_data = json.loads(raw_data)
            task_type = topic.split(":")[-1]
            robot_id = get_best_robot(task_type)
            task_payload = determine_task_payload(task_type, event_data)

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "topic": topic,
                "event": event_data,
                "assigned_robot": robot_id,
                "task_payload": task_payload,
            }

            with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")

            print(
                f"[Dispatcher] Logged task '{task_payload['task_name']}' for global_id "
                f"{event_data.get('global_id')}"
            )
        except (json.JSONDecodeError, KeyError) as parse_exc:
            print(f"[Dispatcher] Failed to parse incoming event: {parse_exc}")
        except Exception as exc:  # Catch-all to keep the dispatcher alive
            print(f"[Dispatcher] Unexpected error: {exc}")


if __name__ == "__main__":
    main()
