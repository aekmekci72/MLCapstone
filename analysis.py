from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

y_pred_proba = model.predict(X_test)
n_classes = len(labels.columns)

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {labels.columns[i]}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="best")
plt.show()

importance_avg = np.mean(importance, axis=1)

plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(importance_avg)[::-1][:20]  
plt.bar(range(len(sorted_idx)), importance_avg[sorted_idx], align='center')
plt.xticks(range(len(sorted_idx)), [mlb.classes_[i] for i in sorted_idx], rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Average Importance')
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.show()
