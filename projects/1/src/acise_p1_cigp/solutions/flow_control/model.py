# S1: best dev program (val fitness 0.249). A change touching old code (or one
# already seen) rarely breaks for the first time; otherwise a long gap since the
# previous build means more unbuilt changes piled up.
THRESHOLDS_S1 = {"gap": ("time_interval", 108.0)}

# S2: S1 plus a slow previous build (pr_duration), the third most used feature.
THRESHOLDS_S2 = {"gap": ("time_interval", 425.0), "slow": ("pr_duration", 1144.31)}

# S3: S2, but a long gap only flags builds with few config-file changes
# (config_files lt motif: on dev, 39% of builds with no config change fail,
# 28% of those with one).
THRESHOLDS_S3 = {
    "gap": ("time_interval", 571.0),
    "slow": ("pr_duration", 1450.69),
    "cfg": ("config_files", 1.0),
}


def predict_s1(b, t) -> bool:
    if b["has_old_change"] >= 1:
        return False
    return b["time_interval"] >= t["gap"]


def predict_s2(b, t) -> bool:
    if b["has_old_change"] >= 1:
        return False
    if b["time_interval"] >= t["gap"]:
        return True
    return b["pr_duration"] >= t["slow"]


def predict_s3(b, t) -> bool:
    if b["has_old_change"] >= 1:
        return False
    if b["time_interval"] >= t["gap"] and b["config_files"] < t["cfg"]:
        return True
    return b["pr_duration"] >= t["slow"]


CANDIDATES = {
    "S1": (predict_s1, THRESHOLDS_S1),
    "S2": (predict_s2, THRESHOLDS_S2),
    "S3": (predict_s3, THRESHOLDS_S3),
}

# Defaults above: tuned on dev fit (dev_split). Dev-val MCC: S1 0.156, S2 0.208,
# S3 0.146; best dev GP program 0.249 (val fitness). Frozen: S2.
predict, THRESHOLDS = CANDIDATES["S2"]
