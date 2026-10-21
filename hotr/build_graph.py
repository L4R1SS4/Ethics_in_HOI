import json
from collections import defaultdict

import networkx as nx


INPUT_FILE = "hoi_triplets_1000.json"
GRAPHS_FILE = "hoi_graphs_1000.json"
SUMMARY_FILE = "hoi_summary_1000.json"

# Limiar para considerar duas detecções da mesma classe como a mesma entidade.
IOU_THRESHOLD = 0.7

# Quantidade de interações (pessoa-ação-objeto) mais frequentes no resumo.
TOP_N = 60


def calculate_iou(box_a, box_b):
    """IoU entre duas caixas [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def get_or_create_node(graph, entities, label, box):
    """
    Reaproveita um nó existente da mesma classe com IoU alto
    (vale para pessoas e objetos); caso contrário, cria um novo.
    """
    best_id, best_iou = None, 0.0

    for entity in entities:
        if entity["label"] != label:
            continue
        iou = calculate_iou(entity["box"], box)
        if iou >= IOU_THRESHOLD and iou > best_iou:
            best_id, best_iou = entity["id"], iou

    if best_id is None:
        count = sum(1 for e in entities if e["label"] == label) + 1
        best_id = "{}_{}".format(label.replace(" ", "_"), count)
        graph.add_node(best_id, label=label, box=box)
        entities.append({"id": best_id, "label": label, "box": box})

    return best_id


def build_graph(image_data):
    """Grafo direcionado pessoa -> objeto, com todas as ações de cada par."""
    graph = nx.DiGraph()
    entities = []

    for interaction in image_data["interactions"]:
        subject = interaction["subject"]
        obj = interaction["object"]
        action = interaction["action"]["label"]
        score = interaction["action"]["score"]

        source = get_or_create_node(graph, entities, subject["label"], subject["box"])
        target = get_or_create_node(graph, entities, obj["label"], obj["box"])

        # Detecções sobrepostas da mesma pessoa não formam interação.
        if source == target:
            continue

        if graph.has_edge(source, target):
            actions = graph[source][target]["actions"]
            actions[action] = max(actions.get(action, 0.0), score)
        else:
            graph.add_edge(source, target, actions={action: score})

    return graph


def graph_to_dict(graph, image_name):
    nodes = [
        {"id": node_id, "label": attrs["label"], "box": attrs["box"]}
        for node_id, attrs in graph.nodes(data=True)
    ]

    edges = [
        {
            "source": source,
            "target": target,
            "actions": [
                {"action": a, "score": s}
                for a, s in sorted(attrs["actions"].items(), key=lambda x: -x[1])
            ],
        }
        for source, target, attrs in graph.edges(data=True)
    ]

    return {"image": image_name, "nodes": nodes, "edges": edges}


def build_summary(graphs):
    """Resumo agregado, compacto, para enviar à LLM."""
    triplet_images = defaultdict(set)   # (ação, objeto) -> imagens
    images_with_interactions = 0
    images_with_2plus_people = 0
    people_total = 0
    people_multi = 0

    for g in graphs:
        labels = {n["id"]: n["label"] for n in g["nodes"]}

        if g["edges"]:
            images_with_interactions += 1

        if sum(1 for lb in labels.values() if lb == "person") >= 2:
            images_with_2plus_people += 1

        per_person = defaultdict(set)

        for edge in g["edges"]:
            object_label = labels[edge["target"]]
            for a in edge["actions"]:
                triplet_images[(a["action"], object_label)].add(g["image"])
                per_person[edge["source"]].add((a["action"], object_label))

        people_total += len(per_person)
        people_multi += sum(1 for v in per_person.values() if len(v) >= 2)

    ranked = sorted(
        triplet_images.items(),
        key=lambda x: (-len(x[1]), x[0]),
    )[:TOP_N]

    return {
        "images_analyzed": len(graphs),
        "images_with_interactions": images_with_interactions,
        "images_with_2plus_people_interacting": images_with_2plus_people,
        "people_with_interactions": people_total,
        "people_with_2plus_distinct_interactions": people_multi,
        "top_interactions": [
            {"person": "person", "action": a, "object": o, "images": len(imgs)}
            for (a, o), imgs in ranked
        ],
    }


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Verifica se o JSON de entrada tem as caixas.
    first = next((i for img in data for i in img["interactions"]), None)
    if first is not None and "box" not in first["subject"]:
        raise SystemExit(
            "O JSON de entrada não tem caixas. "
            "Rode o predictions_to_triplets com KEEP_BOXES = True."
        )

    graphs = [
        graph_to_dict(build_graph(image_data), image_data["image"])
        for image_data in data
    ]

    with open(GRAPHS_FILE, "w") as f:
        json.dump(graphs, f, indent=2)

    summary = build_summary(graphs)

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Imagens:", len(graphs))
    print("Nós:", sum(len(g["nodes"]) for g in graphs))
    print("Arestas:", sum(len(g["edges"]) for g in graphs))
    print("Grafos:", GRAPHS_FILE)
    print("Resumo (para a LLM):", SUMMARY_FILE)


if __name__ == "__main__":
    main()