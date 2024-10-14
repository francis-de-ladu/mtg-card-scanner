from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
from PIL import Image, ImageEnhance

from .enhancements import combined_filters


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


SPACE = " "
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "1234567890"


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_card(image, card_contour):
    pts = card_contour.reshape(-1, 2)
    # rect = cv2.boundingRect(pts)

    # Order points for perspective transform
    pts = order_points(pts)

    (tl, tr, br, bl) = pts

    # Compute width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the "birds eye view"
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Perspective transform
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def extract_cards(mask_generator, frame_path: Path, out_dir: Path) -> list[str]:
    frame = np.array(Image.open(frame_path).convert("RGB"))
    if frame.shape[0] < frame.shape[1]:
        frame = np.rot90(frame, 3)

    if 1080 not in frame.shape:
        resize = 1080 / min(frame.shape[:2])
        frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = mask_generator.generate(frame)

    summary_props = [
        "area",
        "bbox",
        "point_coords",
        "predicted_iou",
        "segmentation",
        "stability_score",
    ]
    masks_summary = [{prop: mask[prop] for prop in summary_props} for mask in masks]

    # print(len(masks))
    # if len(masks) < 8:
    #     pprint(masks_summary)

    # plt.figure(figsize=(6, 6))
    # plt.imshow(frame)
    # show_anns(masks)
    # plt.axis("on")
    # plt.show()

    cards = show_cards(frame, masks_summary, out_dir=out_dir)
    return list(filter(any, cards))  # type: ignore[arg-type]


def show_cards(frame: np.ndarray, masks: list[dict[str, Any]], out_dir: Path) -> list[tuple[str, str]]:
    # # retrieve the mask associated to the card
    # mask = sorted(masks, key=lambda x: x["area"], reverse=True)[1]

    texts = []
    names = []
    for mask in sorted(masks, key=lambda x: x["area"], reverse=True):
        seg = mask["segmentation"]

        contours, _ = cv2.findContours(
            seg.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        card_contours = []

        for contour in contours:
            eps = 0.02
            epsilon = eps * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # print(len(approx))

            if 4 <= len(approx) <= 6:  # Only quadrilateral shapes are considered
                card_contours.append(approx)

        # # Drawing contours on the original image
        # cv2.drawContours(frame, card_contours, -1, (0, 255, 0), 3)

        # plt.figure(figsize=(8, 8))
        # plt.imshow(frame)
        # plt.axis("on")
        # plt.show()

        # # extract the card from the frame
        # x1, y1, w, h = map(int, mask["bbox"])
        # # point_coords = np.array(mask["point_coords"][0]) - [x1, y1]
        # card = frame[y1 : y1 + h, x1 : x1 + w]

        # print((x1, x1 + w), (y1, y1 + h))

        # plt.figure(figsize=(8, 8))
        # plt.imshow(card)
        # # show_points(point_coords, 1, plt.gca())
        # plt.axis("on")
        # plt.show()

        for card_contour in card_contours:
            warped_card = warp_card(frame, card_contour)

            h, w, _ = warped_card.shape
            if 600 < h < frame.shape[0] and 400 < w < frame.shape[0]:
                print((w, h))
            if not (
                (
                    600 <= h <= frame.shape[0]
                    and 400 <= w <= frame.shape[1]
                    and (h < frame.shape[0] or w < frame.shape[1])
                )
                and (1.25 <= h / w <= 2.1)
            ):
                continue

            out_path = out_dir / "warped_card.png"
            out_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(out_path.as_posix(), cv2.cvtColor(warped_card, cv2.COLOR_RGB2BGR))
            # plt.savefig(out_path)
            # plt.imshow(warped_card)
            # plt.show()

            # warped_name = cv2.bitwise_not(warped_card[: h // 11, w // 15 : -w // 4])
            # warped_card = -warped_card[-h // 11 :, : w // 6]
            # warped_name = warped_card[h // 18 : h // 8, w // 15 : -w // 4]
            warped_name = warped_card[h // 18 : h // 8, : -w // 4]
            warped_card = warped_card[-h // 11 :, : w // 5]

            enhanced_name = ImageEnhance.Contrast(
                ImageEnhance.Sharpness(Image.fromarray(warped_name)).enhance(2.5)
            ).enhance(1.5)
            enhanced_name = np.array(enhanced_name.convert("RGB"))  # type: ignore[assignment]
            # enhanced_card = ImageEnhance.Contrast(
            #     ImageEnhance.Sharpness(Image.fromarray(warped_card)).enhance(5)
            # ).enhance(1.5)
            # enhanced_card = np.array(enhanced_card.convert("L"))  # type: ignore[assignment]

            # enhanced_name = combined_filters(warped_name)
            enhanced_card = combined_filters(warped_card)
            enhanced_name = combined_filters(enhanced_name)
            # enhanced_card = combined_filters(enhanced_card)

            # Now you can apply OCR to warped_card
            name = pytesseract.image_to_string(
                enhanced_name,
                # warped_name,
                config=f"-c tessedit_char_whitelist='{UPPER}{LOWER}{SPACE}' preserve_interword_spaces=1 --psm 13",
            )
            names.append(name.replace("\n\n", "\n").strip())

            # Now you can apply OCR to warped_card
            text = pytesseract.image_to_string(
                enhanced_card,
                # warped_card,
                config=f"-c tessedit_char_whitelist='{UPPER}{DIGITS}{SPACE}/' preserve_interword_spaces=1 --psm 6",
            )
            texts.append(text.replace("\n\n", "\n").strip())

            plt.imshow(enhanced_name)
            # plt.imshow(warped_name)
            plt.show()
            plt.imshow(enhanced_card)
            # plt.imshow(warped_card)
            plt.show()

            print((names[-1], texts[-1]))

    return list(zip(names, texts))
