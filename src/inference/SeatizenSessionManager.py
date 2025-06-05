import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
from matplotlib.gridspec import GridSpec

from ..utils.raster_color import RASTER_CLASS_COLOR
class SeatizenSessionManager:

    def __init__(self, session_path: Path) -> None:
        
        self.session_path = session_path
        self.__ortho_path = None
        self.__new_prediction_raster_path = None

    @property
    def orthophoto_path(self) -> Path:
        
        if self.__ortho_path == None:
            self.__ortho_path = Path(self.session_path, "PROCESSED_DATA", "PHOTOGRAMMETRY", "odm_orthophoto", "odm_orthophoto.tif")

            if not self.__ortho_path.exists():
                raise FileNotFoundError(f"{self.__ortho_path} not found.")
        
        return self.__ortho_path
    
    def move_prediction_raster(self, prediction_raster: Path, model_name: str) -> None:

        IA_path = Path(self.session_path, "PROCESSED_DATA", "IA")
        IA_path.mkdir(exist_ok=True, parents=True)

        self.__new_prediction_raster_path = Path(IA_path, f"{self.session_path.name}_{model_name}_ortho_predictions.tif")
        shutil.move(prediction_raster, self.__new_prediction_raster_path)
    
    def create_resume_pdf(self, model_name: str) -> None:
        self.__new_prediction_raster_path = "/media/bioeos/E/drone/serge_temp/20231202_REU-TROU-DEAU_UAV-01_01/PROCESSED_DATA/IA/20231202_REU-TROU-DEAU_UAV-01_01_SegForCoralBig-2025_05_20_27648-bs16_refine_b2_ortho_predictions.tif"
        if self.__new_prediction_raster_path == None:
            print("We don't found the prediction raster path")
            return 
        
        pdf_resume_path = Path(self.session_path, f"000_{self.session_path.name}_preview.pdf")

        # Example coral class mapping
        coral_labels = {
            1: "Acropora Branching",
            2: "Acropora Tabular",
            3: "Non-acropora Massive",
            4: "Others Corals",
            5: "Sand"
        }

        # Load everything
        cmap = ListedColormap([
            np.array(RASTER_CLASS_COLOR.get(i, (0, 0, 0, 255)))[:3] / 255.0
            for i in range(6)
        ])
        rgb, rgb_transform, width, height = self.load_downsampled_rgb_and_transform()
        with rasterio.open(self.orthophoto_path) as ortho_src:
            seg_resampled = self.resample_seg_to_rgb(self.__new_prediction_raster_path, (height, width), rgb_transform, ortho_src.crs)

        figsize = (8.27, 11.69)  # A4 portrait
        figsize_paysage = (11.69, 8.27)  # A4 portrait


        print("func: pred is read")
        # Prepare the figure
        with PdfPages(pdf_resume_path) as pdf:
            
            # Page 1
            fig = plt.figure(figsize=figsize)
            fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
            gs = GridSpec(15, 1, figure=fig)  # 10 rows, 1 column grid

            # Title section
            ax_title = fig.add_subplot(gs[0, 0])
            ax_title.axis('off')
            ax_title.text(0.5, 0.9, "Segmentation Report", ha='center', va='center', fontsize=20)
            ax_title.text(0.5, 0.4, f"Session: {self.session_path.name}", ha='center', va='center', fontsize=10)
            ax_title.text(0.5, 0.1, f"Segmentation model: {model_name}", ha='center', va='center', fontsize=10)

            # Overlay section
            ax_overlay = fig.add_subplot(gs[1:, 0])
            ax_overlay.imshow(rgb)
            ax_overlay.imshow(seg_resampled, cmap=cmap, alpha=0.4, interpolation='none')
            ax_overlay.set_title("Segmentation Overlay")
            ax_overlay.axis('off')

            pdf.savefig(fig)
            plt.close(fig)

            # Page 2
            fig_master = plt.figure(figsize=figsize)
            fig_master.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
            gs = GridSpec(2, 1, figure=fig_master)  # 10 rows, 1 column grid

            # Stats section
            colors = [cmap(int(label)) for label in list(RASTER_CLASS_COLOR)]
            unique, counts = np.unique(seg_resampled, return_counts=True)
            labels = [coral_labels[i] for i in unique[1:]]
            
            ax_stats = fig_master.add_subplot(gs[0, 0])
            ax_stats.axis('off')

            # fig, ax = plt.subplots(figsize=figsize_paysage/2, subplot_kw=dict(aspect="equal"))

            wedges, texts = ax_stats.pie(counts[1:], wedgeprops=dict(width=0.3), startangle=-60, colors=colors)

            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-"),
                    bbox=bbox_props, zorder=0, va="center")

            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = f"angle,angleA=0,angleB={ang}"
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax_stats.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)

            # Overlay section
            ax_legend = fig_master.add_subplot(gs[1, 0])
            for i, (index, label) in enumerate(list(coral_labels.items())[::-1]):
                rgba = np.array(RASTER_CLASS_COLOR.get(index, (0, 0, 0, 255))) / 255.0
                ax_legend.add_patch(plt.Rectangle((0, i), 0.5, 1, color=rgba, ec="black"))
                ax_legend.text(0.6, i + 0.5, label, va='center', fontsize=12)

            ax_legend.set_xlim(0, 2)
            ax_legend.set_ylim(0, len(coral_labels))
            ax_legend.set_xticks([])
            ax_legend.set_yticks([])
            ax_legend.set_frame_on(False)

            ax_legend.axis('off')

            pdf.savefig(fig_master)
            plt.close(fig_master)
            
            # Statistics
            # plt.title("Predicted Class Distribution", y=0.0, pad=0.5)
            # pdf.savefig()
            # plt.close()
            # print("stat")


            # Legend Page
            # legend_fig = self.create_legend_figure(RASTER_CLASS_COLOR, coral_labels, figsize_paysage)
            # plt.title("Segmentation Classes by Color", y=0.0, pad=0.5)
            # pdf.savefig(legend_fig)
            # plt.close(legend_fig)

            # (Optional) Add more pages like close-up views or a legend
            for i, file in enumerate(list(Path("/tmp/00_segment/final/").iterdir())):
                if i > 10: break
                img = mpimg.imread(file)
                fig, ax = plt.subplots(figsize=figsize)
                ax.imshow(img)
                ax.axis('off')  # Hide axes
                pdf.savefig(fig)
                plt.close(fig)
    
    def load_downsampled_rgb_and_transform(self, scale=0.05):
        with rasterio.open(self.orthophoto_path) as src:
            new_height = int(src.height * scale)
            new_width = int(src.width * scale)
            transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            rgb = src.read(
                [1, 2, 3],
                out_shape=(3, new_height, new_width),
                resampling=Resampling.average
            )
            rgb = np.transpose(rgb, (1, 2, 0))  # HWC
            rgb = np.clip(rgb / 255.0, 0, 1)
        return rgb, transform, new_width, new_height

    def resample_seg_to_rgb(self, seg_path, dst_shape, dst_transform, dst_crs):
        with rasterio.open(seg_path) as src:
            src_seg = src.read(1)
            dst_seg = np.zeros(dst_shape, dtype=np.uint8)

            reproject(
                source=src_seg,
                destination=dst_seg,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=WarpResampling.nearest
            )
        return dst_seg
    
    def create_legend_figure(self, color_dict, class_labels, figsize):
        fig, ax = plt.subplots(figsize=figsize/2)

        for i, (index, label) in enumerate(list(class_labels.items())[::-1]):
            rgba = np.array(color_dict.get(index, (0, 0, 0, 255))) / 255.0
            ax.add_patch(plt.Rectangle((0, i), 0.5, 1, color=rgba, ec="black"))
            ax.text(0.6, i + 0.5, label, va='center', fontsize=12)

        ax.set_xlim(0, 2)
        ax.set_ylim(0, len(class_labels))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        return fig