import argparse
import json
import os
from collections import defaultdict

import cv2

IMAGE_DIR = "hico_20160224_det/images/test2015"
TRIPLETS_FILE = "hoi_triplets_1000.json"
OUTPUT_DIR = "visualizations"
SAMPLE_STEP = 10  # mesmo valor usado na amostragem do HOTR

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera a imagem anotada com as interações HOI detectadas."
    )
    parser.add_argument(
        "number",
        type=int,
        help="Número da imagem amostrada (1 a N, em que N é o total de imagens do projeto).",
    )
    return parser.parse_args()


def put_text_lines(image, lines, x, y_bottom, color):
    """Escreve várias linhas empilhadas para cima, terminando em y_bottom."""
    y = y_bottom
    for line in reversed(lines):
        cv2.putText(
            image, line, (x, max(y, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        y -= 22


def main():
    args = parse_args()

    with open(TRIPLETS_FILE, "r") as f:
        data = json.load(f)

    total = len(data)

    if not 1 <= args.number <= total:
        raise SystemExit(
            "Número inválido: {}. Digite um valor entre 1 e {}.".format(args.number, total)
        )

    sample = data[args.number - 1]
    filename = sample["image"]
    original_index = (args.number - 1) * SAMPLE_STEP

    image_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        raise SystemExit("Não foi possível abrir: {}".format(image_path))

    if sample["interactions"] and "box" not in sample["interactions"][0]["subject"]:
        raise SystemExit(
            "O JSON não tem caixas. Rode o predictions_to_triplets com KEEP_BOXES = True."
        )

    # Agrupa as ações de um mesmo par (pessoa, objeto) para desenhar uma vez só.
    pairs = defaultdict(list)
    for interaction in sample["interactions"]:
        subject = interaction["subject"]
        obj = interaction["object"]
        key = (
            tuple(int(v) for v in subject["box"]),
            tuple(int(v) for v in obj["box"]),
            obj["label"],
        )
        pairs[key].append(
            (interaction["action"]["label"], interaction["action"]["score"])
        )

    for (sbox, obox, object_label), actions in pairs.items():
        sx1, sy1, sx2, sy2 = sbox
        ox1, oy1, ox2, oy2 = obox

        cv2.rectangle(image, (sx1, sy1), (sx2, sy2), GREEN, 2)
        cv2.rectangle(image, (ox1, oy1), (ox2, oy2), BLUE, 2)

        cv2.putText(
            image, "person", (sx1, max(sy1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2,
        )

        actions.sort(key=lambda x: -x[1])
        lines = [object_label] + [
            "{} ({:.2f})".format(name, score) for name, score in actions
        ]
        put_text_lines(image, lines, ox1, oy1 - 8, BLUE)

        # Liga o centro da pessoa ao centro do objeto.
        subject_center = ((sx1 + sx2) // 2, (sy1 + sy2) // 2)
        object_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)

        cv2.arrowedLine(
            image, subject_center, object_center, YELLOW, 2, tipLength=0.03
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR, "hoi_{:03d}_{}".format(args.number, filename)
    )
    cv2.imwrite(output_path, image)

    print("Número no projeto:", args.number, "de", total)
    print("Índice no conjunto original:", original_index)
    print("Arquivo original:", image_path)
    print("Pares pessoa-objeto desenhados:", len(pairs))
    print("Imagem anotada salva em:", output_path)

    if not sample["interactions"]:
        print("Aviso: nenhuma interação acima do limiar nesta imagem.")


if __name__ == "__main__":
    main()