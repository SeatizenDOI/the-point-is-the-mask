import shutil
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..PathManager import PathManager
from ..ConfigParser import ConfigParser

from .tools_evaluation import clip_ortho_on_test_zone, convert_tiles_into_png, split_ortho_into_tiles, perform_inference, merge_predictions, resize_merged_raster


def perform_evalutation(pm: PathManager, cp: ConfigParser, model_path: Path) -> None:

    print("\n\n------ [EVALUATION] ------\n")

    if not pm.uav_csv.exists():
        raise FileNotFoundError(f"{pm.uav_csv} not found")

    df_uav_ortho = pd.read_csv(pm.uav_csv)

    id2label = {}
    for i, row in df_uav_ortho.iterrows():
        ortho_path = Path(row["root_folder"], row["ortho_name"])
        if not ortho_path.exists():
            print(f"{ortho_path} not found.")
            continue
        
        print(f"Working with orthophoto: {ortho_path.name}")
        # Pour chaque zone, on fait un dossier final raster et on met toutes les clipped ortho_png dedans
        for drone_test_zone_geojson, annotation_mask_png in cp.drone_zone_polygon_path:
            
            drone_test_zone_geojson_path, annotation_mask_png_path = Path(drone_test_zone_geojson), Path(annotation_mask_png)

            if not drone_test_zone_geojson_path.exists():
                print(f"{drone_test_zone_geojson_path} not found")
                continue

            if not annotation_mask_png_path.exists():
                print(f"{annotation_mask_png_path} not found")
                continue
            
            if pm.eval_tmp_folder.exists():
                shutil.rmtree(pm.eval_tmp_folder)
            pm.eval_tmp_folder.mkdir(exist_ok=True, parents=True)    

            print(f"Working with test_zone: {drone_test_zone_geojson_path.name}")


            # Clipping the ortho.
            clip_ortho_path = Path(pm.eval_tmp_folder, f"{ortho_path.stem}_{drone_test_zone_geojson_path.stem}.tif")
            if not clip_ortho_on_test_zone(ortho_path, drone_test_zone_geojson_path, clip_ortho_path):
                continue

            # We split the orthophoto into tiles.
            eval_tile_folder = Path(pm.eval_tmp_folder, "tiles")
            eval_images_folder = Path(pm.eval_tmp_folder, "images")
            split_ortho_into_tiles(clip_ortho_path, cp.tile_size, cp.horizontal_overlap, cp.vertical_overlap, eval_tile_folder)

            # We convert the tiles into png.
            convert_tiles_into_png(eval_tile_folder, eval_images_folder)
            
            # Perform inference on images.
            eval_count_folder = Path(pm.eval_tmp_folder, "counts")
            id2label = perform_inference(model_path, eval_images_folder, eval_count_folder, cp.base_model_name)
          
            # Merge predictons.
            merged_pred_path = Path(pm.eval_tmp_folder, "full_prediction.tif")
            merge_predictions(clip_ortho_path, eval_count_folder, merged_pred_path, cp.tile_size)
            
            # Resize merged raster.
            output_png_path = Path(pm.eval_prediction_on_annotation_zone, f"{ortho_path.stem}__{drone_test_zone_geojson_path.stem}.png")
            resize_merged_raster(merged_pred_path, annotation_mask_png_path, output_png_path, drone_test_zone_geojson_path)
            

    all_metrics = []
    all_gt_flat = []
    all_pred_flat = []
    all_cm_normalized = {}
    all_label_names = {}

    # === PER-ZONE EVALUATION ===
    for drone_poly, gt_png_path in cp.drone_zone_polygon_path:
        drone_poly_path = Path(drone_poly)
        print(f"\n📊 Evaluating zone: {drone_poly}")
        for pred_png_path in pm.eval_prediction_on_annotation_zone.iterdir():
            if drone_poly_path.stem not in pred_png_path.name: continue


            pred = np.array(Image.open(pred_png_path))
            gt = np.array(Image.open(gt_png_path))

            assert pred.shape == gt.shape, f"Shape mismatch in {drone_poly}: pred {pred.shape}, gt {gt.shape}"

            gt_flat = gt.flatten()
            pred_flat = pred.flatten()
            mask = gt_flat != 0

            gt_flat_masked = gt_flat[mask]
            pred_flat_masked = pred_flat[mask]

            all_gt_flat.append(gt_flat_masked)
            all_pred_flat.append(pred_flat_masked)

            labels = sorted(list(set(np.unique(np.concatenate([gt_flat_masked, pred_flat_masked]))) - {0}))
            label_names = [id2label[lbl] for lbl in labels]

            cm = confusion_matrix(gt_flat_masked, pred_flat_masked, labels=labels)
            cm_normalized = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-7) * 100

            all_cm_normalized[pred_png_path.stem] = cm_normalized
            all_label_names[pred_png_path.stem] = label_names

            # === Metrics ===
            epsilon = 1e-7
            intersection = np.diag(cm)
            per_class_acc = intersection / (np.sum(cm, axis=1) + epsilon)
            per_class_acc_for_mean = per_class_acc[per_class_acc > 0]
            mean_acc = np.mean(per_class_acc_for_mean)
            pixel_acc = np.sum(intersection) / np.sum(cm)


            print(f"  ✅ Pixel Accuracy: {pixel_acc:.4f}")
            print(f"  ✅ Mean Accuracy : {mean_acc:.4f}")   

            print("  Pixel Accuracy Per Class:")
            for i, cls in enumerate(labels):
                print(f"    {id2label[cls]}: {per_class_acc[i]:.4f}")

            all_metrics.append({
                "zone": pred_png_path.stem,
                "mean_acc": mean_acc,   
                "pixel_acc": pixel_acc
            })

    # === MICRO-AVERAGED METRICS ===
    print("\n📦 Micro-Averaged Metrics Across Zones (all pixels):")

    gt_all = np.concatenate(all_gt_flat)
    pred_all = np.concatenate(all_pred_flat)

    labels_all = sorted(list(set(np.unique(np.concatenate([gt_all, pred_all]))) - {0}))
    label_names_all = [id2label[l] for l in labels_all]

    cm = confusion_matrix(gt_all, pred_all, labels=labels_all)
    cm_normalized_all = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-7) * 100

    # Save global confusion matrix for later plotting
    all_cm_normalized["global"] = cm_normalized_all
    all_label_names["global"] = label_names_all

    # === Global Metrics ===
    epsilon = 1e-7
    intersection = np.diag(cm)
    per_class_acc = intersection / (np.sum(cm, axis=1) + epsilon)
    mean_acc = np.mean(per_class_acc)
    pixel_acc = np.sum(intersection) / np.sum(cm)

    print("  Pixel Accuracy Per Class:")
    for i, cls in enumerate(labels):
        print(f"    {id2label[cls]}: {per_class_acc[i]:.4f}")

    print(f"\n  ✅ Pixel Accuracy: {pixel_acc:.4f}")
    print(f"  ✅ Mean Accuracy : {mean_acc:.4f}")


    # === PLOT ALL CONFUSION MATRICES SIDE BY SIDE ===
    print("\n📸 Plotting confusion matrices...")

    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    zones_to_plot = [a.stem for a in pm.eval_prediction_on_annotation_zone.iterdir()] + ["global"]

    for ax, zone in zip(axs, zones_to_plot):
        sns.heatmap(all_cm_normalized[zone], annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=all_label_names[zone], yticklabels=all_label_names[zone],
                    cbar=False, ax=ax)
        ax.set_title(zone.upper())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    outpath = Path(pm.eval_folder, "confusion_matrix_all_in_one.png")
    plt.savefig(outpath)

    # === PLOT GLOBAL CONFUSION MATRIX ONLY ===
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized_all, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=label_names_all, yticklabels=label_names_all,
                cbar_kws={'label': 'Percentage (%)'})
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    outpath = Path(pm.eval_folder, "confusion_matrix.png")
    plt.savefig(outpath, dpi=300)
    plt.close()