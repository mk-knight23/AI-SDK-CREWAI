import argparse

try:
    from .app import run_crewai_mission
except ImportError:
    from app import run_crewai_mission


def demo(mission: str) -> None:
    out = run_crewai_mission(mission)
    print("[CrewAI] primary:", out.get("primary"))
    print("[CrewAI] support:", out.get("support"))
    print("[CrewAI] result:", out.get("result"))
    print("[CrewAI] verification:", out.get("verification"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="build product and run growth launch")
    args = parser.parse_args()
    demo(args.mission)
