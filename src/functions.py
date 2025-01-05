from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
from loguru import logger
from PIL import Image, ImageEnhance
from scipy.cluster.hierarchy import dendrogram, fcluster, ward
from scipy.spatial.distance import pdist, squareform

from .enhancements import combined_filters

MIN_AREA = 700_000


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


def line_intersection(line1, line2):
    """
    Finds the intersection of two lines in 2D space.

    Parameters:
    line1: tuple of two 2D vectors, representing the first line. (p1, p2)
    line2: tuple of two 2D vectors, representing the second line. (p3, p4)

    Returns:
    The intersection point as a 2D numpy array or None if the lines are parallel.
    """
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    p3, p4 = np.array(line2[0]), np.array(line2[1])

    # Compute direction vectors
    d1 = p2 - p1  # Direction of the first line
    d2 = p4 - p3  # Direction of the second line

    # Formulate as A * [t; s] = b
    A = np.array([d1, -d2]).T  # Coefficients matrix
    b = p3 - p1  # Right-hand side vector

    try:
        # Solve for the parameters t and s
        t, s = np.linalg.solve(A, b)
        # Calculate the intersection point using t (parametric representation for line1)
        intersection_point = p1 + t * d1
        return intersection_point
    except np.linalg.LinAlgError:
        # Lines are parallel, no intersection
        return None


def order_points(pts):
    # print()
    print(f"{pts = }")
    if len(pts) > 4:
        nth = len(pts) - 4 - 1
        dists = 2 * pdist(pts, "minkowski", p=-10) ** 1.1 + pdist(pts, "minkowski", p=10) ** 1.1
        # dists = pdist(pts, "minkowski", p=0.5) + pdist(pts, "minkowski", p=10)
        # dists = pdist(pts, "minkowski", p=1)
        # dists = pdist(pts, "euclidean")
        dist_matrix = squareform(dists.round())

        threshold = np.partition(dists, nth)[nth]
        Z = ward(dists)

        clusters = fcluster(Z, threshold, criterion="distance")

        mask = np.zeros_like(clusters).astype(bool)

        cs = Counter(clusters)
        for val, cnt in cs.items():
            if cnt == 2:
                mask[clusters == val] = True

        new_pts = []

        # print(f"{dist_matrix = }")
        for val, cnt in cs.items():
            if cnt == 2:
                # print(f"{val = }")
                cdists = dist_matrix[clusters == val]
                cdists[:, mask] = np.inf
                # print(f"{cdists = }")

                # smallest_idx = np.unravel_index(np.argsort(cdists, axis=None), cdists.shape)
                # smallest_3 = np.array(smallest_idx).T[:3].T.tolist()
                # print(f"{smallest_3 = }")

                cpoints = np.argwhere(clusters == val).flatten().tolist()
                print(f"{cpoints = }")

                nearests = [
                    (cpoints[0] - 1) % len(pts),
                    (cpoints[1] + 1) % len(pts),
                ]

                # nearests = []
                # while len(nearests) < 2:
                #     smallest_idx = np.unravel_index(np.argmin(cdists, axis=None), cdists.shape)
                #     nearests.append(smallest_idx[1])

                #     cdists[smallest_idx[0]] = np.inf
                #     cdists[:, smallest_idx[1]] = np.inf

                # nearests = cdists.argmin(axis=1).tolist())
                # print(f"{nearests = }")

                # print(f"{list(zip(cpoints, nearests)) = }")

                # print(pts[cpoints[0]], pts[nearests[0]])
                # print(pts[cpoints[1]], pts[nearests[1]])

                intersection = line_intersection(
                    [pts[cpoints[0]], pts[nearests[0]]],
                    [pts[cpoints[1]], pts[nearests[1]]],
                )
                print(f"{intersection = }")
                line_pts = np.concatenate([pts[nearests], [intersection]])
                print(f"{line_pts = }")
                supsup = pdist(line_pts, "euclidean")
                print(f"{supsup = }")
                if supsup.min() < 50:
                    continue
                new_pts.append(intersection.round().astype(int))
            else:
                new_pts.append(pts[np.argwhere(clusters == val)[0][0]])

        print(f"{new_pts = }")
        pts = np.asarray(new_pts)

        # plt.figure()
        # dendrogram(Z)
        # plt.show()
        # print(Z)

    dists = pdist(pts, metric="sqeuclidean")
    # print(squareform(dists.round()))

    threshold = np.partition(dists, 1)[1]
    Z = ward(dists)
    clusters = fcluster(Z, threshold, criterion="distance")

    corners = []
    for val in np.unique(clusters):
        cpoints = pts[clusters == val]
        if cpoints[0][0] > cpoints[1][0]:
            cpoints = cpoints[::-1]

        if not corners:
            corners.extend(cpoints.tolist())
        elif np.asarray(corners).min(axis=0)[1] < cpoints.min(axis=0)[1]:
            corners.extend(cpoints.tolist())
        else:
            corners = cpoints.tolist() + corners

    corners = np.asarray(corners)
    # print(corners)

    return corners

    # rect = np.zeros((4, 2), dtype="float32")

    # s = pts.sum(axis=1)
    # print(pts)
    # print(s)
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]

    # diff = np.diff(pts, axis=1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]

    # print(f"{rect = }")
    # return rect


SPACE = " "
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "1234567890"


def warp_card(image, card_contour):
    pts = card_contour.reshape(-1, 2)
    # rect = cv2.minAreaRect(pts)
    # box = cv2.boxPoints(rect)
    # print(f"{rect = }")
    # print(f"{box = }")
    # rect = cv2.boundingRect(pts)

    # Order points for perspective transform
    pts = order_points(pts)
    # pts = box.round().astype(int)
    # print(f"{pts = }")

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    tp = max(0, -ymin)
    bt = max(0, ymax - image.shape[0])
    lt = max(0, -xmin)
    rt = max(0, xmax - image.shape[1])

    image = cv2.copyMakeBorder(image, tp, bt, lt, rt, cv2.BORDER_CONSTANT)
    supsup = cv2.drawContours(image.copy(), [card_contour], -1, (0, 255, 0), 3)
    # plt.imshow(supsup)
    # plt.show()

    (tl, tr, bl, br) = (pts + [lt, tp]).astype(np.float32)

    # Compute width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # maxWidth = (maxHeight * 5 / 7).round().astype(int)

    # Destination points for the "birds eye view"
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]],
        dtype="float32",
    )

    # print(pts)
    # print(dst)

    # Perspective transform
    M = cv2.getPerspectiveTransform(np.asarray([tl, tr, bl, br]), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if maxWidth < maxHeight:
        maxWidth = 5 * maxHeight // 7
    else:
        maxWidth = 7 * maxHeight // 5

    warped = cv2.resize(warped, (maxWidth, maxHeight))

    # plt.imshow(warped)
    # plt.show()

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
    masks_summary = [
        {prop: mask[prop] for prop in summary_props} for mask in masks if mask["area"] > MIN_AREA
    ][:1]

    # print(len(masks))
    # if len(masks) < 8:
    #     pprint(masks_summary)

    plt.figure(figsize=(6, 6))
    plt.imshow(frame)
    # show_anns(masks)
    show_anns(masks_summary)
    plt.axis("on")
    plt.show()

    cards = show_cards(frame, masks_summary, out_dir=out_dir)
    return list(filter(any, cards))  # type: ignore[arg-type]


def show_cards(frame: np.ndarray, masks: list[dict[str, Any]], out_dir: Path) -> list[tuple[str, str]]:
    # # retrieve the mask associated to the card
    # mask = sorted(masks, key=lambda x: x["area"], reverse=True)[1]

    texts = []
    names = []
    for mask in sorted(masks, key=lambda x: x["area"], reverse=True):
        # print(f'{mask["area"] = }')
        if mask["area"] < MIN_AREA:
            continue

        seg = mask["segmentation"]

        contours, _ = cv2.findContours(
            seg.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
            # seg.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            # seg.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )

        card_contours = []

        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue

            for eps in np.arange(0.005, 0.025, 0.0025):
                epsilon = eps * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                # print(len(approx))
                if not cv2.isContourConvex(approx):
                    print(f"{approx = }")
                    approx = cv2.convexHull(approx)

                print(f"{approx = }")

                if 4 <= len(approx) <= 6:  # Only quadrilateral shapes are considered
                    card_contours.append(approx)
                    broke_out = True
                    break

        # print(f"{len(card_contours) = }")
        # print(f"{card_contours = }")

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
            try:
                warped_card = warp_card(frame, card_contour)
            except ValueError as e:
                logger.error(f"Error: {e}")
                continue

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
            plt.imshow(warped_card)
            plt.show()

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

            # # Defining all the parameters
            # t_lower = 15 # Lower Threshold
            # t_upper = 40 # Upper threshold
            # aperture_size = 3 # Aperture size
            # L2Gradient = True # Boolean

            # # Applying the Canny Edge filter
            # # with Aperture Size and L2Gradient
            # edges = cv2.Canny(warped_name, t_lower, t_upper,
            #                 apertureSize = aperture_size,
            #                 L2gradient = L2Gradient )

            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

            gray = cv2.cvtColor(warped_name, cv2.COLOR_BGR2GRAY)
            # gray = cv2.cvtColor(enhanced_name, cv2.COLOR_BGR2GRAY)

            sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

            thresh1 = cv2.threshold(sharpen, 128, 255, cv2.THRESH_TRIANGLE)[1]
            thresh2 = cv2.threshold(sharpen, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)[1]
            thresh3 = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU)[1]
            thresh4 = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # thresh1 = cv2.filter2D(thresh1, -1, sharpen_kernel)
            # thresh2 = cv2.filter2D(thresh2, -1, sharpen_kernel)
            # thresh3 = cv2.filter2D(thresh3, -1, sharpen_kernel)
            # thresh4 = cv2.filter2D(thresh4, -1, sharpen_kernel)

            # thresh1 = cv2.cvtColor(np.array(Image.fromarray(thresh1).convert("RGB")), cv2.COLOR_BGR2GRAY)
            # thresh2 = cv2.cvtColor(np.array(Image.fromarray(thresh2).convert("RGB")), cv2.COLOR_BGR2GRAY)
            # thresh3 = cv2.cvtColor(np.array(Image.fromarray(thresh3).convert("RGB")), cv2.COLOR_BGR2GRAY)
            # thresh4 = cv2.cvtColor(np.array(Image.fromarray(thresh4).convert("RGB")), cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
            thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
            thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR)
            thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR)

            thresh1 = cv2.filter2D(thresh1, -1, sharpen_kernel)
            thresh2 = cv2.filter2D(thresh2, -1, sharpen_kernel)
            thresh3 = cv2.filter2D(thresh3, -1, sharpen_kernel)
            thresh4 = cv2.filter2D(thresh4, -1, sharpen_kernel)

            plt.grid(False)
            plt.axis("off")
            plt.subplot(811), plt.imshow(warped_name, cmap="gray")
            plt.subplot(812), plt.imshow(enhanced_name, cmap="gray")
            plt.subplot(813), plt.imshow(gray, cmap="gray")
            plt.subplot(814), plt.imshow(sharpen)  # s, cmap="gray")
            plt.subplot(815), plt.imshow(thresh1)  # , cmap="gray")
            plt.subplot(816), plt.imshow(thresh2)  # , cmap="gray")
            plt.subplot(817), plt.imshow(thresh3)  # , cmap="gray")
            plt.subplot(818), plt.imshow(thresh4)  # , cmap="gray")
            plt.grid(False)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            # Now you can apply OCR to warped_card
            name = pytesseract.image_to_string(
                # enhanced_name,
                warped_name,
                config=f"-c tessedit_char_whitelist='{UPPER}{LOWER}{SPACE}' preserve_interword_spaces=1 --psm 13",
                # config=f"-c tessedit_char_whitelist='{UPPER}{LOWER}{SPACE}' preserve_interword_spaces=1 --psm 6",
            )
            names.append(name.replace("\n\n", "\n").strip())

            for tresh in (thresh1, thresh2, thresh3, thresh4):
                name = pytesseract.image_to_string(
                    # enhanced_name,
                    tresh,
                    config=f"-c tessedit_char_whitelist='{UPPER}{LOWER}{SPACE}' preserve_interword_spaces=1 --psm 13",
                    # config=f"-c tessedit_char_whitelist='{UPPER}{LOWER}{SPACE}' preserve_interword_spaces=1 --psm 6",
                )
                print(f"{name = }")
                names.append(name.replace("\n\n", "\n").strip())

            # Now you can apply OCR to warped_card
            text = pytesseract.image_to_string(
                # enhanced_card,
                warped_card,
                config=f"-c tessedit_char_whitelist='{UPPER}{DIGITS}{SPACE}/' preserve_interword_spaces=1 --psm 6",
            )
            texts.append(text.replace("\n\n", "\n").strip())

            plt.imshow(warped_name)
            plt.show()
            # plt.imshow(enhanced_name)
            # plt.show()

            plt.imshow(warped_card)
            plt.show()
            # plt.imshow(enhanced_card)
            # plt.show()

            print(names)
            print((names[-1], texts[-1]))

    return list(zip(names, texts))
