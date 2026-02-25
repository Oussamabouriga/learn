```
tol = 1.0  # 1 point tolerance

acc_train_tol = (np.abs(y_train - y_pred_train) <= tol).mean() * 100
acc_test_tol = (np.abs(y_test - y_pred_test) <= tol).mean() * 100

print(f"Accuracy within ±{tol} point (train): {acc_train_tol:.2f}%")
print(f"Accuracy within ±{tol} point (test) : {acc_test_tol:.2f}%")

```
