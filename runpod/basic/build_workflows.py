"""Prepare native ComfyUI presets from the archived official H3 graphs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
REF = "minimax_h3_ref2va_pruned_bf16.safetensors"
FL = "minimax_h3_fl2va_pruned_bf16.safetensors"
SINGULARITY = "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_bf16.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT = (
    "Photoreal live-action, one continuous shot at normal real-time speed. "
    "Use <Picture 1> for the adult performer's identity and clothing. "
    "Full-body view in a concrete training courtyard under soft daylight. "
    "The performer takes three fast running steps, vaults a low concrete barrier, "
    "lands with both feet and bent knees, then takes two recovery steps. "
    "Keep the whole body and its ground contact visible. Preserve body proportions, "
    "clothing, environment and direction of travel throughout the shot. "
    "Show weight transfer, believable momentum and a clear landing response. "
    "The camera tracks smoothly from the side. Audio: synchronized footsteps, "
    "clothing movement, landing impact and natural outdoor ambience. "
    "No dialogue, music, slow motion, cuts, captions or logos."
)


def node(graph, kind):
    matches = [n for n in graph["nodes"] if n["type"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected one {kind}, found {len(matches)}")
    return matches[0]


def link_parts(link):
    if isinstance(link, list):
        return list(link)
    return [link[k] for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")]


def rebuild(graph):
    """Canonicalize both endpoint records; no dangling graph links are allowed."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    for n in nodes.values():
        for item in n.get("inputs", []):
            item["link"] = None
        for item in n.get("outputs", []):
            item["links"] = []
        n.pop("widgets_values_named", None)
        n.get("properties", {}).pop("models", None)
    seen = set()
    for raw in graph["links"]:
        ident, origin, out_slot, target, in_slot, kind = link_parts(raw)
        if ident in seen or origin not in nodes or target not in nodes:
            raise ValueError(f"Invalid graph link {raw}")
        seen.add(ident)
        source = nodes[origin]["outputs"][out_slot]
        dest = nodes[target]["inputs"][in_slot]
        if dest["link"] is not None:
            raise ValueError(f"Two links target {target}:{in_slot}")
        source["links"].append(ident)
        dest["link"] = ident
        if source["type"] != kind and source["type"] != "*":
            raise ValueError(f"Source type differs for {raw}")
        accepted = dest["type"].split(",")
        if kind not in accepted and "*" not in accepted:
            raise ValueError(f"Target type differs for {raw}")
    graph["links"] = [link_parts(link) for link in graph["links"]]
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = max(seen, default=0)
    graph.pop("definitions", None)
    return graph


def new_link(graph, origin, out_slot, target, in_slot, kind):
    ident = max((link_parts(x)[0] for x in graph["links"]), default=0) + 1
    graph["links"].append([ident, origin, out_slot, target, in_slot, kind])


def set_file(graph, kind, filename, selector=None):
    targets = [n for n in graph["nodes"] if n["type"] == kind]
    if selector:
        targets = [n for n in targets if selector(n)]
    for n in targets:
        n["widgets_values"][0] = filename
        if kind in ("UNETLoader", "CLIPLoader", "VAELoader"):
            n["inputs"] = []


def finish(graph, name):
    graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "h3-basic:" + name))
    graph["revision"] = 0
    node(graph, "SaveVideo")["widgets_values"][0] = "H3_TEST/" + name
    graph["extra"] = {"frontendVersion": "1.55.17", "h3_basic": {
        "name": name, "seed": 42, "turbo": False,
        "comparison_note": "Compare 01 and 05 using the same uploaded image and unchanged prompt/settings.",
        "runtime_gpu_verified": False,
    }}
    return rebuild(graph)


def reference_graph(source):
    g = copy.deepcopy(source)
    removed = {139, 141, 142, 144, 145, 146}
    g["nodes"] = [n for n in g["nodes"] if n["id"] not in removed and n["type"] != "MarkdownNote"]
    g["links"] = [link_parts(x) for x in g["links"] if link_parts(x)[1] not in removed and link_parts(x)[3] not in removed]
    # Replace only the optional Turbo switches with the original full-step path.
    new_link(g, 127, 0, 126, 0, "MODEL")
    new_link(g, 143, 0, 124, 1, "INT")
    set_file(g, "UNETLoader", REF)
    set_file(g, "CLIPLoader", ENCODER)
    set_file(g, "VAELoader", VIDEO_VAE, lambda n: "video_vae" in n["widgets_values"][0])
    set_file(g, "VAELoader", AUDIO_VAE, lambda n: "audio_vae" in n["widgets_values"][0])
    node(g, "ResolutionSelector")["widgets_values"] = ["16:9 (Widescreen)", 1.0, 32]
    node(g, "PrimitiveStringMultiline")["widgets_values"] = [PROMPT]
    node(g, "RandomNoise")["widgets_values"] = [42, "fixed"]
    node(g, "LoadImage")["widgets_values"] = ["UPLOAD_REFERENCE.png", "image"]
    node(g, "LoadImage")["title"] = "Envie a imagem de referencia — mesma imagem em 01 e 05"
    return rebuild(g)


def first_last_graph(source):
    definition = source["definitions"]["subgraphs"][0]
    removed = {121, 122, 123, 125, 126}
    g = {k: copy.deepcopy(v) for k, v in source.items() if k not in ("nodes", "links", "definitions", "groups")}
    g["nodes"] = [copy.deepcopy(n) for n in definition["nodes"] if n["id"] not in removed and n["type"] != "MarkdownNote"]
    g["links"] = [link_parts(x) for x in definition["links"] if link_parts(x)[1] >= 0 and link_parts(x)[3] >= 0 and link_parts(x)[1] not in removed and link_parts(x)[3] not in removed]
    top = {n["id"]: n for n in source["nodes"]}
    first = copy.deepcopy(top[114])
    first["id"] = 200
    first["title"] = "Envie o primeiro frame"
    first["widgets_values"] = ["UPLOAD_FIRST_FRAME.png", "image"]
    last = copy.deepcopy(first)
    last["id"] = 201
    last["title"] = "Envie o ultimo frame"
    last["widgets_values"] = ["UPLOAD_LAST_FRAME.png", "image"]
    size = copy.deepcopy(top[115])
    size["widgets_values"] = ["16:9 (Widescreen)", 1.0, 32]
    save = copy.deepcopy(top[92])
    prompt = copy.deepcopy(node(reference_graph(json.loads((ROOT / "upstream/official-ref2va.json").read_text())), "PrimitiveStringMultiline"))
    prompt["id"] = 202
    prompt["widgets_values"] = [PROMPT.replace("Use <Picture 1> for the adult performer's identity and clothing. ", "Preserve the adult performer's identity and clothing from the input frames. ")]
    g["nodes"].extend([first, last, size, save, prompt])
    for args in [(6, 0, 16, 0, "MODEL"), (6, 0, 9, 0, "MODEL"), (124, 0, 9, 1, "INT"),
                 (200, 0, 104, 2, "IMAGE"), (201, 0, 104, 3, "IMAGE"), (202, 0, 104, 4, "STRING"),
                 (115, 0, 104, 5, "INT"), (115, 1, 104, 6, "INT"), (91, 0, 92, 0, "VIDEO")]:
        new_link(g, *args)
    set_file(g, "UNETLoader", FL)
    set_file(g, "CLIPLoader", ENCODER)
    set_file(g, "VAELoader", VIDEO_VAE, lambda n: "video_vae" in n["widgets_values"][0])
    set_file(g, "VAELoader", AUDIO_VAE, lambda n: "audio_vae" in n["widgets_values"][0])
    node(g, "RandomNoise")["inputs"] = []
    node(g, "RandomNoise")["widgets_values"] = [42, "fixed"]
    node(g, "PrimitiveFloat")["inputs"] = []
    node(g, "PrimitiveFloat")["widgets_values"] = [5]
    node(g, "BasicScheduler")["widgets_values"][1] = 20
    g["groups"] = []
    # Keep native graph nodes visible instead of relying on proxy widgets.
    positions = {6: [-1500, -850], 13: [-1500, -650], 11: [-1500, -450], 24: [-1500, -300],
                 200: [-1500, -100], 201: [-1500, 400], 202: [-950, 700], 115: [-950, -100],
                 111: [-950, 150], 107: [-650, 150], 104: [-400, 100], 124: [0, -800],
                 9: [350, -700], 15: [350, -450], 16: [350, -200], 17: [350, 0],
                 14: [750, 0], 10: [1100, -150], 23: [1100, 150], 91: [1450, 0], 92: [1800, 0]}
    for n in g["nodes"]:
        n["pos"] = positions.get(n["id"], [0, 0])
    return rebuild(g)


def action_graph(base):
    g = copy.deepcopy(base)
    loader = node(g, "UNETLoader")
    consumers = [x for x in g["links"] if x[1] == loader["id"] and x[5] == "MODEL"]
    g["links"] = [x for x in g["links"] if x not in consumers]
    previous = loader["id"]
    for offset, (filename, strength) in enumerate([("H3_Combat_V2.safetensors", 0.8), ("Motion_Repair_V2.safetensors", 0.6)]):
        ident = max(n["id"] for n in g["nodes"]) + 1
        g["nodes"].append({"id": ident, "type": "LoraLoaderModelOnly", "pos": [loader["pos"][0] + 700, loader["pos"][1] + offset * 180],
            "size": [420, 82], "flags": {}, "order": 0, "mode": 0,
            "inputs": [{"name": "model", "type": "MODEL", "link": None}],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
            "properties": {"cnr_id": "comfy-core", "Node name for S&R": "LoraLoaderModelOnly"},
            "widgets_values": [filename, strength]})
        new_link(g, previous, 0, ident, 0, "MODEL")
        previous = ident
    for _, _, out_slot, target, in_slot, kind in consumers:
        new_link(g, previous, out_slot, target, in_slot, kind)
    node(g, "PrimitiveInt")["widgets_values"] = [30, "fixed"]
    node(g, "BasicScheduler")["widgets_values"][1] = 30
    return rebuild(g)


def build():
    sources = []
    raw = []
    for filename, expected in [("official-ref2va.json", "7ffa22d98706bcd5448c09c548108a548529ed8f"),
                               ("official-fl2va.json", "bcc680e74537bffa89bff3eb6eb11a7c220e38e9")]:
        payload = (ROOT / "upstream" / filename).read_bytes()
        actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if actual != expected:
            raise ValueError(f"Archived upstream Git blob differs: {filename}: {actual}")
        raw.append(json.loads(payload))
        sources.append({"file": filename, "git_blob_sha": actual})
    ref = reference_graph(raw[0])
    fl = first_last_graph(raw[1])
    singularity = copy.deepcopy(ref)
    set_file(singularity, "UNETLoader", SINGULARITY)
    presets = [("01_REF_Oficial_BF16", ref), ("02_FirstLast_Oficial_BF16", fl),
               ("03_REF_Acao_BF16", action_graph(ref)), ("04_FirstLast_Acao_BF16", action_graph(fl)),
               ("05_REF_Singularity_INT8_Comparacao", singularity)]
    output = ROOT / "workflows"
    output.mkdir(exist_ok=True)
    records = []
    for name, graph in presets:
        payload = (json.dumps(finish(graph, name), ensure_ascii=False, indent=2) + "\n").encode()
        (output / (name + ".json")).write_bytes(payload)
        records.append({"file": name + ".json", "sha256": hashlib.sha256(payload).hexdigest()})
    (ROOT / "workflows.provenance.json").write_text(json.dumps({"upstream": sources,
        "repository": "https://github.com/Comfy-Org/workflow_templates", "presets": records,
        "adaptations": ["Selected model/encoder/VAEs", "Removed optional Turbo paths", "1 MP horizontal and fixed seed 42",
                        "Single reference input; exposed first and last frame", "Action-only LoRA chains and 30 steps"],
        "official_originals_preserved": True, "runtime_gpu_verified": False}, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
