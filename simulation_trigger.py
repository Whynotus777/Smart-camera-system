import json
import subprocess
import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0


def connect_to_redis() -> redis.Redis:
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    client.ping()
    print(f"[Simulation] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return client


def run_isaac_sim_scenario(event_data: dict) -> None:
    reason = event_data.get("reason", "unknown")
    print(f"[SIMULATOR] Received trigger for event: {reason}. Starting sim job...")
    # TODO: This function will launch an Isaac Sim instance.
    # 1. Load the 7-Eleven Digital Twin (e.g., 'store.usd').
    # 2. Place a virtual robot (e.g., Carter) at its charging dock.
    # 3. Use `event_data` to "re-enact" the scene:
    #    - Spawn a virtual human actor.
    #    - Use `event_data['bbox']` and `event_data['camera']` to map to a 3D location.
    #    - Command the actor to perform the 'concealing' or 'spill' animation.
    # 4. Command the virtual robot GPRM with a text prompt (e.g., "A person is shoplifting, go observe").
    # 5. Run the simulation and record the robot's [Vision, Language, Action] data to disk for VLA model training.
    subprocess.run(["echo", "Simulation job stub finished."], check=False)


def main():
    try:
        redis_client = connect_to_redis()
    except Exception as exc:
        print(f"[Simulation] Failed to connect to Redis: {exc}")
        return

    pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
    pubsub.psubscribe("store:events:*")
    print("[Simulation] Listening for events on pattern 'store:events:*'")

    for message in pubsub.listen():
        if message["type"] not in {"pmessage", "message"}:
            continue

        try:
            raw_data = message.get("data")
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")

            event_data = json.loads(raw_data)
            run_isaac_sim_scenario(event_data)
        except (json.JSONDecodeError, KeyError) as parse_exc:
            print(f"[Simulation] Failed to parse incoming event: {parse_exc}")
        except Exception as exc:
            print(f"[Simulation] Unexpected error: {exc}")


if __name__ == "__main__":
    main()
