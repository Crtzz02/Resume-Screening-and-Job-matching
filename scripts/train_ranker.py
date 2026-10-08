"""
Training script for the supervised match ranker.

Current Hybrid Formula:
hybrid_score = (0.4 * tfidf_score + 0.6 * semantic_score) / (0.4 + 0.6)
where semantic_score is from all-MiniLM-L6-v2 embeddings, and tfidf_score is from n-grams with skill weights.

This script uses the current matching functions to generate weak labels and extract features, 
then trains a Logistic Regression ranker to learn a better combination.
"""
import os
import sys
import json
import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# Ensure the root path is added so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.field_catalog import load_job_fields
from src.extract import extract_text_from_bytes, SUPPORTED_EXTENSIONS
from src.preprocessing import clean_text
from src.matching import compute_similarity, compute_semantic_similarity
from src.ranker import extract_features_dict

def load_resumes(resume_dir: str) -> Dict[str, str]:
    resumes = {}
    if not os.path.exists(resume_dir):
        print(f"Warning: Resume directory not found at {resume_dir}")
        return resumes
        
    for filename in os.listdir(resume_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            filepath = os.path.join(resume_dir, filename)
            with open(filepath, "rb") as f:
                raw_bytes = f.read()
            try:
                text = extract_text_from_bytes(filename, raw_bytes)
                if text.strip():
                    resumes[filename] = text
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    return resumes

def generate_dataset() -> Tuple[pd.DataFrame, bool]:
    # 1. Load data
    resume_dir = os.path.join(project_root, "data", "resumes")
    fields = load_job_fields()
    resumes = load_resumes(resume_dir)
    
    print(f"Loaded {len(resumes)} resumes and {len(fields)} fields.")
    
    rows = []
    
    # Check if gold labels exist
    gold_path = os.path.join(project_root, "data", "ranker_labels.csv")
    has_gold = os.path.exists(gold_path) and os.path.getsize(gold_path) > 0
    gold_df = None
    if has_gold:
        gold_df = pd.read_csv(gold_path)
        print("Using Gold Labels from data/ranker_labels.csv")
    else:
        print("Using Weak Labels based on current hybrid scoring.")
    
    for res_id, res_text in resumes.items():
        res_clean = clean_text(res_text)
        res_semantic = " ".join(res_text.lower().split())
        res_len = len(res_clean.split())
        
        scores = []
        for field in fields:
            field_clean = clean_text(field.match_text)
            field_semantic = " ".join(field.match_text.lower().split())
            
            tfidf_score, _ = compute_similarity(res_clean, field_clean, skill_weights=field.skill_weights)
            tfidf_score = float(tfidf_score) if not np.isnan(tfidf_score) else 0.0
            
            semantic_score, _ = compute_semantic_similarity(res_semantic, field_semantic)
            semantic_score = float(semantic_score) if not np.isnan(semantic_score) else 0.0
            
            hybrid_score = (0.4 * tfidf_score + 0.6 * semantic_score)
            
            matched_skills = [
                skill for skill in field.key_skills
                if skill.lower() in res_text.lower() or skill.lower() in res_clean
            ]
            missing_skills = [s for s in field.key_skills if s not in matched_skills]
            
            features = extract_features_dict(
                tfidf_score=tfidf_score,
                semantic_score=semantic_score,
                hybrid_score=hybrid_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                key_skills=field.key_skills,
                skill_weights=field.skill_weights,
                resume_len=res_len,
                field_len=len(field_clean.split())
            )
            
            features["resume_id"] = res_id
            features["field_id"] = field.id
            features["field_name"] = field.title
            
            scores.append(features)
            
        # Weak labeling logic
        if not has_gold:
            scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
            top_score = scores[0]["hybrid_score"]
            for i, row in enumerate(scores):
                # Positive if top-1 or within 0.05 of top-1 and score > 0.15
                if (i == 0 or (top_score - row["hybrid_score"] <= 0.05)) and row["hybrid_score"] > 0.15:
                    row["label"] = 1
                else:
                    row["label"] = 0
                rows.append(row)
        else:
            rows.extend(scores)
            
    df = pd.DataFrame(rows)
    
    if has_gold and gold_df is not None:
        # The CSV has ['filename', 'true_field'] which correspond to resume_id and field_id
        # We assign label=1 if the pair exists in the CSV, else label=0
        gold_positives = set(zip(gold_df['filename'], gold_df['true_field']))
        
        def assign_label(row):
            return 1 if (row['resume_id'], row['field_id']) in gold_positives else 0
            
        df['label'] = df.apply(assign_label, axis=1)
        
    return df, has_gold

def main():
    df, has_gold = generate_dataset()
    
    # Features to use for training
    feature_names = [
        "tfidf_score", "semantic_score", "hybrid_score", 
        "matched_skill_count", "missing_skill_count", 
        "skill_coverage", "weighted_skill_score",
        "resume_length", "field_length"
    ]
    
    if df.empty:
        print("Error: No data available for training.")
        return

    X = df[feature_names].values
    y = df["label"].values
    
    print(f"\nDataset shape: {df.shape}, Positives: {y.sum()}")
    if y.sum() == 0 or y.sum() == len(y):
        print("Error: Labels have only one class. Cannot train model. Try adjusting weak label margins.")
        return

    # Train model
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    
    # Track top-1 and top-3 for both baseline and model
    baseline_top1, baseline_top3 = 0, 0
    model_top1, model_top3 = 0, 0
    
    # Fit overall model to save
    model.fit(X, y)
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    df["ranker_proba"] = y_pred_proba
    
    # Threshold for point metrics
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    try:
        final_roc_auc = roc_auc_score(y, y_pred_proba)
        final_pr_auc = average_precision_score(y, y_pred_proba)
        final_f1 = f1_score(y, y_pred)
        final_precision = precision_score(y, y_pred)
        final_recall = recall_score(y, y_pred)
        
        print("\n=== Ranker Evaluation ===")
        print(f"ROC-AUC:   {final_roc_auc:.4f}")
        print(f"PR-AUC:    {final_pr_auc:.4f}")
        print(f"F1 Score:  {final_f1:.4f}")
        print(f"Precision: {final_precision:.4f} (at {threshold})")
        print(f"Recall:    {final_recall:.4f} (at {threshold})")
    except Exception as e:
        print(f"Could not compute some metrics: {e}")

    # Evaluate Hit Rate (Top-1, Top-3) per resume (only counting where label==1 exists)
    eval_resumes = 0
    for res_id, group in df.groupby("resume_id"):
        positives = group[group["label"] == 1]["field_id"].tolist()
        if not positives:
            continue
            
        eval_resumes += 1
        
        # Baseline top
        base_top = group.sort_values("hybrid_score", ascending=False)["field_id"].tolist()
        if base_top[0] in positives:
            baseline_top1 += 1
        if any(p in base_top[:3] for p in positives):
            baseline_top3 += 1
            
        # Model top
        mod_top = group.sort_values("ranker_proba", ascending=False)["field_id"].tolist()
        if mod_top[0] in positives:
            model_top1 += 1
        if any(p in mod_top[:3] for p in positives):
            model_top3 += 1

    if eval_resumes > 0:
        print(f"\n=== Hit Rate Comparison (over {eval_resumes} evaluated resumes) ===")
        print(f"Baseline (hybrid) Top-1: {baseline_top1/eval_resumes:.4f}")
        print(f"Ranker (learned)  Top-1: {model_top1/eval_resumes:.4f}")
        print(f"Baseline (hybrid) Top-3: {baseline_top3/eval_resumes:.4f}")
        print(f"Ranker (learned)  Top-3: {model_top3/eval_resumes:.4f}")
        
        if model_top1 < baseline_top1 and model_top3 < baseline_top3:
            print("\nWARNING: The trained model does NOT beat the baseline on hit rates.")
            print("You may need more data, better weak labels, or gold labels.")
        else:
            print("\nSUCCESS: The ranker matches or beats the baseline hit rates!")
    
    # Save Model
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "hybrid_ranker.joblib")
    meta_path = os.path.join(model_dir, "ranker_meta.json")
    
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    meta_info = {
        "model_type": "LogisticRegression",
        "train_date": datetime.datetime.now().isoformat(),
        "label_source": "gold" if has_gold else "weak",
        "feature_names": feature_names,
        "metrics": {
            "roc_auc": float(final_roc_auc) if 'final_roc_auc' in locals() else None,
            "pr_auc": float(final_pr_auc) if 'final_pr_auc' in locals() else None,
            "f1": float(final_f1) if 'final_f1' in locals() else None,
            "precision": float(final_precision) if 'final_precision' in locals() else None,
            "recall": float(final_recall) if 'final_recall' in locals() else None,
            "baseline_top1": baseline_top1/eval_resumes if eval_resumes else 0,
            "model_top1": model_top1/eval_resumes if eval_resumes else 0,
            "baseline_top3": baseline_top3/eval_resumes if eval_resumes else 0,
            "model_top3": model_top3/eval_resumes if eval_resumes else 0
        },
        "coefficients": dict(zip(feature_names, model.coef_[0].tolist()))
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    print(f"Meta info saved to {meta_path}")

if __name__ == "__main__":
    main()
