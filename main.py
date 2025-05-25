from fairface_eval import load_fairface_model, evaluate_folder, kl_divergence
from mitigation_pipeline import gender_balancing_filter
from clip_eval import get_clip_embeddings, diversity_score
from blip_eval import generate_caption
from utils import list_images, plot_distribution
from fairface_eval import classify_demographics, load_fairface_model
import os

if __name__ == "__main__":
    gen_path = "generated_images"
    model = load_fairface_model()

    gender_counts, race_counts = evaluate_folder(model, gen_path)
    print("Gender Counts:", gender_counts)
    print("Race Counts:", race_counts)

    # KL 计算示例
    fairface_gender_dist = [0.5, 0.5]  # 理想平衡
    pred_gender_dist = [gender_counts['Male'], gender_counts['Female']]
    print("KL Divergence (Gender):", kl_divergence(pred_gender_dist, fairface_gender_dist))

    # Diversity
    embeddings = get_clip_embeddings(gen_path)
    print("Visual Diversity Score:", diversity_score(embeddings))

    # Caption 示例
    sample = list_images(gen_path)[0]
    print("Caption for", sample, ":", generate_caption(sample))

    # 画图
    plot_distribution(gender_counts, title="Gender Distribution")

