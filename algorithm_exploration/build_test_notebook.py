import json
import os
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))

scaling_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 10. Uji Skalabilitas: Waktu Pemrosesan vs Jumlah Data\n",
        "\n",
        "Bagian ini menguji ketiga model Deep Learning (**ArcFace**, **Facenet**, **VGG-Face**) pada lima ukuran dataset yang berbeda: 100, 300, 500, 700, dan 1000 gambar (`test_case_100` s.d. `test_case_700`, serta `test_case` untuk 1000 gambar). Untuk setiap ukuran data, dicatat waktu **deteksi & alignment wajah** serta waktu **ekstraksi fitur** per model, tanpa menggunakan cache, agar waktu yang terukur murni mencerminkan biaya komputasi untuk jumlah data tersebut."
    ]
}

scaling_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "SCALING_SIZES = [100, 300, 500, 700, 1000]\n",
        "SCALING_MODELS = [\"ArcFace\", \"Facenet\", \"VGG-Face\"]\n",
        "\n",
        "def get_scaling_dir(size):\n",
        "    if size == 1000:\n",
        "        return os.path.join(TESTSET_DIR, \"test_case\")\n",
        "    return os.path.join(TESTSET_DIR, f\"test_case_{size}\")\n",
        "\n",
        "scaling_results = {\n",
        "    \"Data Size\": [],\n",
        "    \"Model\": [],\n",
        "    \"Num Images\": [],\n",
        "    \"Num Faces Detected\": [],\n",
        "    \"Detection Time (s)\": [],\n",
        "    \"Feature Extraction Time (s)\": [],\n",
        "    \"Total Time (s)\": []\n",
        "}"
    ]
}

scaling_run_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import gc\n",
        "\n",
        "for size in SCALING_SIZES:\n",
        "    img_dir = get_scaling_dir(size)\n",
        "    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n",
        "    print(f\"\\n=== Ukuran Data: {size} ({len(img_files)} gambar ditemukan di {os.path.basename(img_dir)}) ===\")\n",
        "\n",
        "    # Deteksi & alignment wajah sekali per ukuran data, dipakai bersama oleh ketiga model\n",
        "    t0 = time.time()\n",
        "    scaling_faces = []\n",
        "    for img_name in img_files:\n",
        "        img_path = os.path.join(img_dir, img_name)\n",
        "        rois = detector.detect_and_crop(img_path)\n",
        "        for roi in rois:\n",
        "            scaling_faces.append(roi[\"cropped_face\"])\n",
        "    det_time = time.time() - t0\n",
        "    print(f\"Deteksi: {len(scaling_faces)} wajah dalam {det_time:.2f} detik\")\n",
        "\n",
        "    for model_name in SCALING_MODELS:\n",
        "        t0 = time.time()\n",
        "        for face_img in scaling_faces:\n",
        "            _ = extract_deepface_embedding(face_img, model_name)\n",
        "        feat_time = time.time() - t0\n",
        "\n",
        "        scaling_results[\"Data Size\"].append(size)\n",
        "        scaling_results[\"Model\"].append(model_name)\n",
        "        scaling_results[\"Num Images\"].append(len(img_files))\n",
        "        scaling_results[\"Num Faces Detected\"].append(len(scaling_faces))\n",
        "        scaling_results[\"Detection Time (s)\"].append(det_time)\n",
        "        scaling_results[\"Feature Extraction Time (s)\"].append(feat_time)\n",
        "        scaling_results[\"Total Time (s)\"].append(det_time + feat_time)\n",
        "\n",
        "        print(f\"  {model_name}: ekstraksi {feat_time:.2f} detik (total {det_time + feat_time:.2f} detik)\")\n",
        "        gc.collect()\n",
        "\n",
        "    del scaling_faces\n",
        "    gc.collect()\n",
        "\n",
        "import tensorflow as tf\n",
        "tf.keras.backend.clear_session()\n",
        "gc.collect()"
    ]
}

scaling_plot_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "df_scaling = pd.DataFrame(scaling_results)\n",
        "print(\"\\n--- Hasil Uji Skalabilitas (Waktu vs Jumlah Data) ---\")\n",
        "print(df_scaling.to_string(index=False))\n",
        "\n",
        "scaling_colors = {'ArcFace': '#4E79A7', 'Facenet': '#F28E2B', 'VGG-Face': '#E15759'}\n",
        "\n",
        "plt.figure(figsize=(14, 6))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "for model_name in SCALING_MODELS:\n",
        "    subset = df_scaling[df_scaling[\"Model\"] == model_name]\n",
        "    plt.plot(subset[\"Data Size\"], subset[\"Total Time (s)\"], marker='o', label=model_name, color=scaling_colors[model_name])\n",
        "plt.title(\"Waktu Total (Deteksi + Ekstraksi) vs Jumlah Data\", fontsize=12, fontweight=\"bold\")\n",
        "plt.xlabel(\"Jumlah Gambar\")\n",
        "plt.ylabel(\"Waktu Total (detik)\")\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "for model_name in SCALING_MODELS:\n",
        "    subset = df_scaling[df_scaling[\"Model\"] == model_name]\n",
        "    plt.plot(subset[\"Data Size\"], subset[\"Feature Extraction Time (s)\"], marker='o', label=model_name, color=scaling_colors[model_name])\n",
        "plt.title(\"Waktu Ekstraksi Fitur vs Jumlah Data\", fontsize=12, fontweight=\"bold\")\n",
        "plt.xlabel(\"Jumlah Gambar\")\n",
        "plt.ylabel(\"Waktu Ekstraksi (detik)\")\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "assets_dir = os.path.join(project_root, \"assets\")\n",
        "os.makedirs(assets_dir, exist_ok=True)\n",
        "plt.savefig(os.path.join(assets_dir, f\"uji_skalabilitas_{DETECTOR_TYPE}.png\"), dpi=300, bbox_inches=\"tight\")\n",
        "plt.show()"
    ]
}


def build_test_notebook(detector_type):
    src_path = os.path.join(current_dir, f"notebook_{detector_type}.ipynb")
    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb["cells"][0]["source"][0] = (
        f"# Eksplorasi Algoritma Face Recognition ({detector_type.upper()} Detector) "
        "+ Uji Skalabilitas\n"
    )

    detector_type_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"DETECTOR_TYPE = \"{detector_type}\"\n"]
    }

    nb["cells"].extend([
        detector_type_cell,
        scaling_markdown_cell,
        scaling_setup_cell,
        scaling_run_cell,
        scaling_plot_cell,
    ])

    out_path = os.path.join(current_dir, f"notebook_{detector_type}_test.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Jupyter Notebook uji skalabilitas ({detector_type}) berhasil dibentuk di: {out_path}")


build_test_notebook("mtcnn")
build_test_notebook("scrfd")
