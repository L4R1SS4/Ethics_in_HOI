import json
import torch

from hotr.data.datasets import builtin_meta

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PREDICTIONS_FILE = "hotr_predictions_1000.pt"
ANNOTATION_FILE = "hico_20160224_det/annotations/test_hico.json"  # ajuste o caminho
ACTION_FILE = "hico_20160224_det/list_action.txt"
OUTPUT_FILE = "hoi_triplets_1000.json"

SAMPLE_STEP = 10        # mesmo valor usado no HICODetection
THRESHOLD = 0.25
MAX_PER_IMAGE = 20      # limite de interações por imagem (mais confiáveis)
KEEP_BOXES = True
NO_INTERACTION = "no_interaction"

# ============================================================
# MAPEAMENTOS
# ============================================================

meta = builtin_meta._get_coco_instances_meta()
object_names = meta["thing_classes"]

action_names = []
with open(ACTION_FILE, "r") as f:
    for line in f.readlines()[2:]:
        parts = line.split()
        if len(parts) >= 2:
            action_names.append(parts[1])

print("Classes de objetos:", len(object_names))
print("Ações:", len(action_names))

# ============================================================
# NOMES DOS ARQUIVOS (mesma amostragem do dataset)
# ============================================================

with open(ANNOTATION_FILE, "r") as f:
    annotations = json.load(f)

sampled_ids = list(range(0, len(annotations), SAMPLE_STEP))

predictions = torch.load(PREDICTIONS_FILE, map_location="cpu")
print("Imagens com predição:", len(predictions))

assert len(predictions) == len(sampled_ids), (
    f"Predições ({len(predictions)}) != imagens amostradas ({len(sampled_ids)}). "
    "Confira SAMPLE_STEP e o arquivo de anotações."
)

# ============================================================
# CONVERSÃO
# ============================================================

all_images = []

for image_idx, prediction in enumerate(predictions):

    labels = prediction["labels"]
    boxes = prediction["boxes"]
    verb_scores = prediction["verb_scores"]
    sub_ids = prediction["sub_ids"]
    obj_ids = prediction["obj_ids"]

    # chave (sujeito, objeto, ação) -> interação com maior score
    best = {}

    for pair_idx in range(len(sub_ids)):
        sub_id = int(sub_ids[pair_idx])
        obj_id = int(obj_ids[pair_idx])

        # sujeito deve ser person
        if int(labels[sub_id]) != 0:
            continue

        object_label_id = int(labels[obj_id])
        if object_label_id < 0 or object_label_id >= len(object_names):
            continue
        object_name = object_names[object_label_id]

        scores = verb_scores[pair_idx]

        # todas as ações acima do limiar (HICO-DET é multirrótulo)
        candidate_ids = torch.nonzero(scores >= THRESHOLD).flatten().tolist()

        for action_id in candidate_ids:
            if action_id >= len(action_names):
                continue

            action_name = action_names[action_id]
            if action_name == NO_INTERACTION:
                continue

            score = float(scores[action_id])
            key = (sub_id, obj_id, action_id)

            if key in best and best[key]["action"]["score"] >= score:
                continue

            interaction = {
                "subject": {"label": "person"},
                "action": {"label": action_name, "score": round(score, 3)},
                "object": {"label": object_name},
            }
            if KEEP_BOXES:
                interaction["subject"]["box"] = boxes[sub_id].tolist()
                interaction["object"]["box"] = boxes[obj_id].tolist()

            best[key] = interaction

    interactions = sorted(
        best.values(),
        key=lambda x: x["action"]["score"],
        reverse=True,
    )[:MAX_PER_IMAGE]

    all_images.append({
        "image": annotations[sampled_ids[image_idx]]["file_name"],
        "interactions": interactions,
    })

# ============================================================
# SALVAR
# ============================================================

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_images, f, indent=2)

total = sum(len(img["interactions"]) for img in all_images)
print("Arquivo:", OUTPUT_FILE)
print("Total de interações:", total)