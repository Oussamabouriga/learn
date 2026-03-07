```
# ==============================
# SHAP global (FIXED for shape mismatch)
# ==============================

import numpy as np
import shap

def _get_class_shap_matrix(shap_values, class_idx: int, X):
    """
    Returns a 2D matrix (n_samples, n_features) for the selected class.
    Handles:
      - shap_values as list (one array per class)
      - shap_values as 3D array (n_samples, n_features, n_classes)
      - extra bias/constant column (n_features + 1) -> drops last col
    """
    # Case 1: list-of-arrays (common in multiclass)
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]  # (n_samples, n_features) or (n_samples, n_features+1)
    else:
        sv = np.asarray(shap_values)
        # Case 2: 3D array
        if sv.ndim == 3:
            # could be (n_samples, n_features, n_classes)
            if sv.shape[0] == X.shape[0] and sv.shape[1] in [X.shape[1], X.shape[1] + 1]:
                sv = sv[:, :, class_idx]
            # or (n_classes, n_samples, n_features)
            elif sv.shape[1] == X.shape[0] and sv.shape[2] in [X.shape[1], X.shape[1] + 1]:
                sv = sv[class_idx, :, :]
            else:
                raise ValueError(f"Unsupported SHAP shape {sv.shape} vs X {X.shape}")
        else:
            # Binary can sometimes be (n_samples, n_features) already
            pass

    sv = np.asarray(sv)

    # Drop extra bias column if present
    if sv.shape[1] == X.shape[1] + 1:
        sv = sv[:, :-1]

    # Final safety check
    if sv.shape[1] != X.shape[1]:
        raise ValueError(f"SHAP shape {sv.shape} does not match X shape {X.shape}")

    return sv


# ---- Compute SHAP values (as you already do)
# shap_values_global = explainer_cls.shap_values(X_shap)

# Choose which class to display
class_for_global = int(pred_example_class)  # or set manually: 0..4

sv_class = _get_class_shap_matrix(shap_values_global, class_for_global, X_shap)

print("\nSHAP global (summary) — shown for class:", class_for_global)
shap.summary_plot(sv_class, X_shap, show=True)

print("\nSHAP global (bar) — shown for class:", class_for_global)
shap.summary_plot(sv_class, X_shap, plot_type="bar", show=True)

```
