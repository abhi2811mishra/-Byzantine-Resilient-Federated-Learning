import glob, json, os

data = {}
for p in sorted(glob.glob("d:/byzantine_fl/results/*.json")):
    try:
        r = json.load(open(p))
        if "attack" in r and "accuracy" in r:
            k = f"{r['attack']}__{r['aggregator']}__{r['byz_frac']}__{r['partition']}"
            data[k] = {
                "acc": [round(x, 4) for x in r["accuracy"]],
                "final": round(r["accuracy"][-1], 4),
                "best": round(max(r["accuracy"]), 4)
            }
    except Exception as e:
        pass

print(f"Extracted {len(data)} runs into web_data.json")
with open("d:/byzantine_fl/web_data.js", "w") as f:
    f.write("const EXPERIMENT_DATA = " + json.dumps(data) + ";\n")

